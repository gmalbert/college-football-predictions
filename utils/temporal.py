"""Point-in-time joins, rolling features, and walk-forward split helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from utils.contracts import ensure_utc, safe_merge


@dataclass(frozen=True)
class TemporalFold:
    fold: int
    train_seasons: tuple[int, ...]
    test_season: int
    train_index: np.ndarray
    test_index: np.ndarray


def to_team_game_long(games: pd.DataFrame) -> pd.DataFrame:
    """Convert one-row-per-game results to one row per participating team."""
    required = {
        "game_id", "season", "week", "start_date", "home_team", "away_team",
    }
    missing = sorted(required.difference(games.columns))
    if missing:
        raise KeyError(f"games frame is missing columns: {missing}")

    base = games.copy()
    base["start_date"] = ensure_utc(base["start_date"])
    if base["game_id"].duplicated().any():
        raise ValueError("to_team_game_long requires one row per game_id")

    home = pd.DataFrame(
        {
            "game_id": base["game_id"],
            "season": base["season"],
            "week": base["week"],
            "start_date": base["start_date"],
            "team": base["home_team"],
            "opponent": base["away_team"],
            "side": "home",
            "points_for": pd.to_numeric(base.get("home_score"), errors="coerce"),
            "points_against": pd.to_numeric(base.get("away_score"), errors="coerce"),
        }
    )
    away = pd.DataFrame(
        {
            "game_id": base["game_id"],
            "season": base["season"],
            "week": base["week"],
            "start_date": base["start_date"],
            "team": base["away_team"],
            "opponent": base["home_team"],
            "side": "away",
            "points_for": pd.to_numeric(base.get("away_score"), errors="coerce"),
            "points_against": pd.to_numeric(base.get("home_score"), errors="coerce"),
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["margin"] = long["points_for"] - long["points_against"]
    long["game_total"] = long["points_for"] + long["points_against"]
    long["win"] = np.where(
        long["margin"].isna(), np.nan, (long["margin"] > 0).astype(float)
    )
    return long.sort_values(["start_date", "game_id", "side"]).reset_index(drop=True)


def add_rest_features(games: pd.DataFrame, *, cap_days: int = 60) -> pd.DataFrame:
    """Add rest using each team's previous game, regardless of home/away side."""
    result = games.copy()
    long = to_team_game_long(result)
    long = long.sort_values(["team", "start_date", "game_id"])
    long["previous_game_at"] = long.groupby("team")["start_date"].shift(1)
    long["rest_days"] = (
        long["start_date"] - long["previous_game_at"]
    ).dt.total_seconds() / 86400.0
    long["rest_days"] = long["rest_days"].clip(lower=0, upper=cap_days)

    home = long[long["side"] == "home"][["game_id", "rest_days"]].rename(
        columns={"rest_days": "rest_days_home"}
    )
    away = long[long["side"] == "away"][["game_id", "rest_days"]].rename(
        columns={"rest_days": "rest_days_away"}
    )
    result = result.drop(
        columns=["rest_days_home", "rest_days_away", "rest_advantage"],
        errors="ignore",
    )
    result = safe_merge(result, home, on="game_id", validate="one_to_one")
    result = safe_merge(result, away, on="game_id", validate="one_to_one")
    result["rest_advantage"] = result["rest_days_home"] - result["rest_days_away"]
    return result


def rolling_team_features(
    team_games: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    windows: Sequence[int] = (3, 5, 8),
    min_periods: int = 1,
    prefix: str = "",
) -> pd.DataFrame:
    """Build pre-game rolling means; the current game is always shifted out."""
    required = {"game_id", "team", "season", "start_date"}
    missing = sorted(required.difference(team_games.columns))
    if missing:
        raise KeyError(f"team_games is missing columns: {missing}")
    values = [column for column in value_columns if column in team_games.columns]
    if not values:
        return team_games[list(required)].copy()

    ordered = team_games.copy()
    ordered["start_date"] = ensure_utc(ordered["start_date"])
    ordered = ordered.sort_values(["team", "season", "start_date", "game_id"])
    if ordered.duplicated(["game_id", "team"]).any():
        raise ValueError("rolling_team_features requires one row per game and team")

    parts: list[pd.DataFrame] = []
    for (_, _), group in ordered.groupby(["team", "season"], sort=False):
        current = group[["game_id", "team", "season", "start_date"]].copy()
        numeric = group[values].apply(pd.to_numeric, errors="coerce").shift(1)
        for window in windows:
            rolled = numeric.rolling(window, min_periods=min_periods).mean()
            for column in values:
                current[f"{prefix}{column}_l{window}"] = rolled[column].to_numpy()
        parts.append(current)
    return pd.concat(parts, ignore_index=True)


def attach_team_features(
    games: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Attach a unique team-game feature table to home and away game sides."""
    if features.duplicated(["game_id", "team"]).any():
        raise ValueError("features must be unique on (game_id, team)")
    ids = {"game_id", "team"}
    chosen = (
        [c for c in feature_columns if c in features.columns]
        if feature_columns is not None
        else [c for c in features.columns if c not in ids | {"season", "start_date"}]
    )
    result = games.copy()
    home = features[["game_id", "team", *chosen]].rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in chosen}}
    )
    away = features[["game_id", "team", *chosen]].rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in chosen}}
    )
    result = safe_merge(
        result, home, on=["game_id", "home_team"], validate="one_to_one"
    )
    result = safe_merge(
        result, away, on=["game_id", "away_team"], validate="one_to_one"
    )
    return result


def point_in_time_join(
    observations: pd.DataFrame,
    history: pd.DataFrame,
    *,
    by: str | Sequence[str],
    observation_time: str,
    available_time: str,
    columns: Sequence[str] | None = None,
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Backward as-of join that cannot select a future feature observation."""
    keys = [by] if isinstance(by, str) else list(by)
    left = observations.copy()
    right = history.copy()
    left[observation_time] = ensure_utc(left[observation_time])
    right[available_time] = ensure_utc(right[available_time])
    selected = list(dict.fromkeys([*keys, available_time, *(columns or [])]))
    right = right[selected]
    left = left.sort_values([*keys, observation_time])
    right = right.sort_values([*keys, available_time])
    merged = pd.merge_asof(
        left,
        right,
        left_on=observation_time,
        right_on=available_time,
        by=keys,
        direction="backward",
        allow_exact_matches=True,
        tolerance=tolerance,
    )
    invalid = merged[available_time] > merged[observation_time]
    if invalid.fillna(False).any():
        raise AssertionError("point-in-time join selected a future row")
    return merged.sort_index()


def walk_forward_season_splits(
    frame: pd.DataFrame,
    *,
    season_col: str = "season",
    time_col: str = "start_date",
    min_train_seasons: int = 2,
    embargo_days: int = 0,
) -> Iterator[TemporalFold]:
    """Yield expanding train seasons and one untouched test season."""
    if season_col not in frame.columns:
        raise KeyError(f"missing season column: {season_col}")
    seasons = sorted(int(value) for value in frame[season_col].dropna().unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError("not enough seasons for a walk-forward split")
    times = ensure_utc(frame[time_col]) if time_col in frame.columns else None

    fold_number = 0
    for offset in range(min_train_seasons, len(seasons)):
        test_season = seasons[offset]
        train_seasons = tuple(seasons[:offset])
        train_mask = frame[season_col].isin(train_seasons)
        test_mask = frame[season_col].eq(test_season)
        if embargo_days and times is not None and test_mask.any():
            test_start = times[test_mask].min()
            train_mask &= times < (test_start - pd.Timedelta(days=embargo_days))
        train_index = np.flatnonzero(train_mask.to_numpy())
        test_index = np.flatnonzero(test_mask.to_numpy())
        if not len(train_index) or not len(test_index):
            continue
        yield TemporalFold(
            fold=fold_number,
            train_seasons=train_seasons,
            test_season=test_season,
            train_index=train_index,
            test_index=test_index,
        )
        fold_number += 1


def availability_audit(
    frame: pd.DataFrame,
    *,
    prediction_time: str,
    feature_times: Sequence[str],
) -> pd.DataFrame:
    """Summarize missing and future-dated values for each source timestamp."""
    if prediction_time not in frame.columns:
        raise KeyError(f"missing prediction timestamp: {prediction_time}")
    predicted = ensure_utc(frame[prediction_time])
    rows = []
    for column in feature_times:
        if column not in frame.columns:
            rows.append(
                {"timestamp": column, "present": False, "missing": len(frame), "future": 0}
            )
            continue
        available = ensure_utc(frame[column])
        rows.append(
            {
                "timestamp": column,
                "present": True,
                "missing": int(available.isna().sum()),
                "future": int((available > predicted).fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows)
