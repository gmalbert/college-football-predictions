"""utils/feature_engine.py

Build the game-level feature matrix by joining all processed Parquet tables.
Saves to data_files/features/feature_matrix.parquet.

Call build_feature_matrix() after fetch_historical.run() completes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.storage import FEATURES_DIR, load_parquet, save_parquet
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────── feature column lists ─────────────────────────

WIN_FEATURES = [
    # Ratings & power
    "elo_diff",
    "sp_plus_diff",
    "sp_offense_diff",
    "sp_defense_diff",
    # Season-level efficiency diffs
    "off_epa_diff",
    "def_epa_diff",
    "off_explosiveness_diff",
    "def_havoc_diff",
    "off_success_diff",
    # Rushing / passing diffs
    "off_rushing_epa_diff",
    "off_passing_epa_diff",
    # Recruiting & talent
    "recruiting_diff",
    "talent_diff",
    "recruiting_rank_diff",
    # Schedule & context
    "home_flag",
    "conference_game",
    "rest_advantage",
    # Rolling 5-game stats (populated after team_game_stats.parquet is built)
    "turnover_margin_l5",
    "rushing_yards_diff_l5",
    "pass_yards_diff_l5",
    "penalty_yards_diff_l5",
]

SPREAD_FEATURES = WIN_FEATURES + ["market_spread"]

TOTAL_FEATURES = [
    "home_off_epa",
    "away_off_epa",
    "home_def_epa",
    "away_def_epa",
    "home_off_explosiveness",
    "away_off_explosiveness",
    "home_off_rushing_epa",
    "away_off_rushing_epa",
    "home_off_passing_epa",
    "away_off_passing_epa",
    "home_flag",
    "rest_days_home",
    "rest_days_away",
    "market_total",
]

ALL_FEATURE_COLS = list(
    dict.fromkeys(WIN_FEATURES + SPREAD_FEATURES + TOTAL_FEATURES)
)

# ───────────────────────────────── helpers ─────────────────────────────────

def _rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days of rest for home and away teams from start_date."""
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.sort_values(["season", "week"])
    for side in ("home", "away"):
        prev = df.groupby(f"{side}_team")["_dt"].shift(1)
        df[f"rest_days_{side}"] = (df["_dt"] - prev).dt.days
    df["rest_advantage"] = df["rest_days_home"] - df["rest_days_away"]
    return df.drop(columns=["_dt"])


def _rolling_team_stats(
    games: pd.DataFrame,
    tgs: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Compute per-team rolling-window stat averages using pre-game history only.
    Returns (team, season, week, {stat}_l{window}) with no data leakage —
    the current game is excluded via .shift(1).
    """
    gw = games[["game_id", "week"]].drop_duplicates()
    tgs = tgs.merge(gw, on="game_id", how="left").dropna(subset=["week"])
    stat_cols = [c for c in tgs.columns
                 if c not in ("game_id", "season", "week", "team", "home_away")]
    parts: list[pd.DataFrame] = []
    for (team, season), grp in tgs.groupby(["team", "season"]):
        grp = grp.sort_values("week").reset_index(drop=True)
        rolled = (
            grp[stat_cols]
            .apply(pd.to_numeric, errors="coerce")
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )
        rolled.columns = [f"{c}_l{window}" for c in stat_cols]
        rolled["team"]   = team
        rolled["season"] = int(season)
        rolled["week"]   = grp["week"].values
        parts.append(rolled)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# ─────────────────────────────── main builder ────────────────────────────────

def build_feature_matrix(force: bool = False) -> pd.DataFrame:
    """
    Join processed tables into a game-level feature matrix.
    Returns the DataFrame and writes it to features/feature_matrix.parquet.
    """
    dst = FEATURES_DIR / "feature_matrix.parquet"
    if not force and dst.exists():
        logger.info("feature_matrix.parquet found — loading cached version")
        return pd.read_parquet(dst)

    # ── load tables ─────────────────────────────────────────────────────────
    try:
        games      = load_parquet("games")
        lines      = load_parquet("lines")
        ratings    = load_parquet("ratings")
        adv        = load_parquet("advanced_stats")
        elo_raw    = load_parquet("elo_ratings")
        recruiting = load_parquet("recruiting")
    except FileNotFoundError as exc:
        logger.error(
            f"Missing processed data ({exc}). "
            "Run `python -m utils.fetch_historical` first."
        )
        return pd.DataFrame()

    # Optional: per-game team stats for rolling features
    try:
        tgs = load_parquet("team_game_stats")
    except FileNotFoundError:
        tgs = None

    # ── Elo: season-level ratings (CFBD v5 provides end-of-season only) ──────
    elo = elo_raw[["season", "team", "elo"]].copy()

    # ── Recruiting: 3-year rolling talent average ────────────────────────────
    recruiting = recruiting.sort_values(["team", "season"])
    recruiting["points_3yr"] = (
        recruiting.groupby("team")["points"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    rec_cols = [c for c in ["season", "team", "points_3yr", "rank"] if c in recruiting.columns]
    rec = recruiting[rec_cols]

    # ── Start building from games ────────────────────────────────────────────
    df = games.copy()

    # Lines
    df = df.merge(lines, on="game_id", how="left")

    # SP+ / talent (home)
    sp_cols = [c for c in ["season", "team", "sp_overall", "sp_offense", "sp_defense", "talent"]
               if c in ratings.columns]
    rat = ratings[sp_cols]
    df = df.merge(
        rat.rename(columns={"team": "home_team", "sp_overall": "home_sp_plus",
                            "sp_offense": "home_sp_offense", "sp_defense": "home_sp_defense",
                            "talent": "home_talent"}),
        on=["season", "home_team"], how="left",
    )
    # SP+ / talent (away)
    df = df.merge(
        rat.rename(columns={"team": "away_team", "sp_overall": "away_sp_plus",
                            "sp_offense": "away_sp_offense", "sp_defense": "away_sp_defense",
                            "talent": "away_talent"}),
        on=["season", "away_team"], how="left",
    )

    # Advanced stats (home + away) — expand to include explosiveness, havoc, rushing/passing EPA
    adv_base = ["season", "team", "off_epa", "def_epa", "off_success_rate", "def_success_rate"]
    adv_ext  = ["off_explosiveness", "def_explosiveness", "def_havoc", "off_havoc",
                "off_rushing_epa", "off_passing_epa", "def_rushing_epa", "def_passing_epa"]
    adv_cols = [c for c in adv_base + adv_ext if c in adv.columns]
    adv_rename_home = {
        "team":              "home_team",
        "off_epa":           "home_off_epa",
        "def_epa":           "home_def_epa",
        "off_success_rate":  "home_off_success",
        "def_success_rate":  "home_def_success",
        "off_explosiveness": "home_off_explosiveness",
        "def_explosiveness": "home_def_explosiveness",
        "def_havoc":         "home_def_havoc",
        "off_havoc":         "home_off_havoc",
        "off_rushing_epa":   "home_off_rushing_epa",
        "off_passing_epa":   "home_off_passing_epa",
        "def_rushing_epa":   "home_def_rushing_epa",
        "def_passing_epa":   "home_def_passing_epa",
    }
    adv_rename_away = {
        "team":              "away_team",
        "off_epa":           "away_off_epa",
        "def_epa":           "away_def_epa",
        "off_success_rate":  "away_off_success",
        "def_success_rate":  "away_def_success",
        "off_explosiveness": "away_off_explosiveness",
        "def_explosiveness": "away_def_explosiveness",
        "def_havoc":         "away_def_havoc",
        "off_havoc":         "away_off_havoc",
        "off_rushing_epa":   "away_off_rushing_epa",
        "off_passing_epa":   "away_off_passing_epa",
        "def_rushing_epa":   "away_def_rushing_epa",
        "def_passing_epa":   "away_def_passing_epa",
    }
    df = df.merge(
        adv[adv_cols].rename(columns=adv_rename_home),
        on=["season", "home_team"], how="left",
    )
    # Advanced stats (away)
    df = df.merge(
        adv[adv_cols].rename(columns=adv_rename_away),
        on=["season", "away_team"], how="left",
    )

    # Elo (home)
    df = df.merge(
        elo.rename(columns={"team": "home_team", "elo": "home_elo"}),
        on=["season", "home_team"], how="left",
    )
    # Elo (away)
    df = df.merge(
        elo.rename(columns={"team": "away_team", "elo": "away_elo"}),
        on=["season", "away_team"], how="left",
    )

    # Recruiting (home)
    df = df.merge(
        rec.rename(columns={"team": "home_team", "points_3yr": "home_recruiting",
                            "rank": "home_recruiting_rank"}),
        on=["season", "home_team"], how="left",
    )
    # Recruiting (away)
    df = df.merge(
        rec.rename(columns={"team": "away_team", "points_3yr": "away_recruiting",
                            "rank": "away_recruiting_rank"}),
        on=["season", "away_team"], how="left",
    )

    # ── derived features ─────────────────────────────────────────────────────
    df["elo_diff"]               = df["home_elo"].fillna(1500) - df["away_elo"].fillna(1500)
    df["sp_plus_diff"]           = df["home_sp_plus"]          - df["away_sp_plus"]
    df["sp_offense_diff"]        = df["home_sp_offense"]       - df["away_sp_offense"]
    # sp_defense: lower = better → away−home positive = home defense advantage
    df["sp_defense_diff"]        = df["away_sp_defense"]       - df["home_sp_defense"]
    df["off_epa_diff"]           = df["home_off_epa"]          - df["away_off_epa"]
    # def_epa: lower (more negative) = better; positive diff = home D advantage
    df["def_epa_diff"]           = df["away_def_epa"]          - df["home_def_epa"]
    df["off_explosiveness_diff"] = df["home_off_explosiveness"]- df["away_off_explosiveness"]
    df["def_havoc_diff"]         = df["home_def_havoc"]        - df["away_def_havoc"]
    df["off_success_diff"]       = df["home_off_success"]      - df["away_off_success"]
    df["off_rushing_epa_diff"]   = df["home_off_rushing_epa"]  - df["away_off_rushing_epa"]
    df["off_passing_epa_diff"]   = df["home_off_passing_epa"]  - df["away_off_passing_epa"]
    df["recruiting_diff"]        = df["home_recruiting"]       - df["away_recruiting"]
    df["talent_diff"]            = df["home_talent"]           - df["away_talent"]
    # recruiting rank: lower number = better → away_rank−home_rank positive = home ranked higher
    if "home_recruiting_rank" in df.columns and "away_recruiting_rank" in df.columns:
        df["recruiting_rank_diff"] = df["away_recruiting_rank"] - df["home_recruiting_rank"]

    # home_flag: 1 = home, 0.5 = neutral site
    df["home_flag"]       = np.where(df["neutral_site"].fillna(False), 0.5, 1.0)
    df["conference_game"] = df["conference_game"].fillna(False).astype(float)

    # ── Rest days ─────────────────────────────────────────────────────────────
    if "start_date" in df.columns:
        df = _rest_days(df)

    # ── Rolling team game stats (optional) ────────────────────────────────────
    if tgs is not None and not tgs.empty:
        rolling = _rolling_team_stats(df, tgs, window=5)
        if not rolling.empty:
            want = ["turnovers", "rushing_yards", "pass_yards", "penalty_yards"]
            keep = ["team", "season", "week"] + [
                f"{s}_l5" for s in want if f"{s}_l5" in rolling.columns
            ]
            rolling = rolling[[c for c in keep if c in rolling.columns]]
            home_r = rolling.rename(columns={"team": "home_team", **{
                f"{s}_l5": f"home_{s}_l5" for s in want}})
            away_r = rolling.rename(columns={"team": "away_team", **{
                f"{s}_l5": f"away_{s}_l5" for s in want}})
            df = df.merge(home_r, on=["home_team", "season", "week"], how="left")
            df = df.merge(away_r, on=["away_team", "season", "week"], how="left")
            if "home_turnovers_l5"     in df.columns and "away_turnovers_l5"     in df.columns:
                df["turnover_margin_l5"]    = df["away_turnovers_l5"]    - df["home_turnovers_l5"]
            if "home_rushing_yards_l5" in df.columns and "away_rushing_yards_l5" in df.columns:
                df["rushing_yards_diff_l5"] = df["home_rushing_yards_l5"] - df["away_rushing_yards_l5"]
            if "home_pass_yards_l5"    in df.columns and "away_pass_yards_l5"    in df.columns:
                df["pass_yards_diff_l5"]    = df["home_pass_yards_l5"]    - df["away_pass_yards_l5"]
            if "home_penalty_yards_l5" in df.columns and "away_penalty_yards_l5" in df.columns:
                df["penalty_yards_diff_l5"] = df["away_penalty_yards_l5"] - df["home_penalty_yards_l5"]

    # ── save ─────────────────────────────────────────────────────────────────
    save_parquet(df, "feature_matrix", layer="features")
    logger.info(
        f"feature_matrix.parquet: {len(df):,} rows × {len(df.columns)} columns"
    )
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build feature matrix")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    args = parser.parse_args()
    build_feature_matrix(force=args.force)
