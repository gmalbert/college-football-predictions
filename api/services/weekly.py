"""Weekly Predictions — mirrors ``pages/1_Weekly_Predictions.py``."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.columns import FEATURE_MATRIX_COLUMNS, LINE_SNAPSHOT_COLUMNS
from api.data import FEATURES_DIR, PROCESSED_DIR, artifact, parquet
from api.jsonutil import jsonable, records
from api.services.common import resolve_timezone
from utils.betting import (
    CONFIDENCE_EMOJI,
    Confidence,
    generate_moneyline_pick,
    generate_spread_pick,
    generate_total_pick,
)
from utils.models import load_metrics, models_trained, predict_for_display
from utils.odds_ingestion import build_market_consensus_from_snapshots
from utils.storage import FEATURES_DIR


def load_feature_matrix() -> pd.DataFrame:
    try:
        return parquet(
            "feature_matrix", layer="features", columns=FEATURE_MATRIX_COLUMNS
        )
    except FileNotFoundError:
        return pd.DataFrame()


def load_market_snapshots() -> pd.DataFrame:
    """Retained market quotes for display only — never model inputs."""
    try:
        snapshots = parquet("line_snapshots", columns=LINE_SNAPSHOT_COLUMNS)
    except FileNotFoundError:
        return pd.DataFrame()
    if snapshots.empty:
        return pd.DataFrame()
    snapshots = snapshots.copy()
    snapshots["captured_at"] = pd.to_datetime(
        snapshots["captured_at"], utc=True, errors="coerce"
    )
    return snapshots.dropna(subset=["game_id", "captured_at"])


def predictions_for(df_all: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Run model inference for one (season, week), memoised on the artifacts.

    Without this the win/spread/total models are re-scored on every request and
    on every slider change.  The result depends only on the feature matrix, the
    saved backtest and the selected week, so it is safe to cache until one of
    those files changes.
    """
    frame = df_all[(df_all["season"] == season) & (df_all["week"] == week)].copy()

    def _compute() -> pd.DataFrame:
        return predict_for_display(frame) if models_trained() else frame

    return artifact(
        f"weekly:predict:{season}:{week}",
        FEATURES_DIR / "feature_matrix.parquet",
        _compute,
    )


def season_consensus(df_all: pd.DataFrame, season: int) -> pd.DataFrame:
    """Display-only market consensus for a whole season, indexed by game_id.

    ``build_market_consensus_from_snapshots`` walks every retained quote and
    groups by game, which cost 110-245 ms per request.  The consensus for a
    given game does not depend on which week or conference filter is selected,
    so it is computed once per (season, snapshot snapshot) and sliced afterwards.
    """
    empty = pd.DataFrame()
    try:
        snapshots = load_market_snapshots()
    except FileNotFoundError:
        return empty
    if snapshots.empty:
        return empty

    schedule = (
        df_all.loc[df_all["season"].eq(season), ["game_id"]]
        .drop_duplicates()
        .assign(season=season)
    )
    if schedule.empty:
        return empty

    consensus = artifact(
        f"weekly:consensus:{season}",
        PROCESSED_DIR / "line_snapshots.parquet",
        lambda: build_market_consensus_from_snapshots(
            snapshots, schedule, season=int(season), exclude_sources=()
        ),
    )
    if consensus is None or consensus.empty:
        return empty
    return consensus.set_index("game_id")


def _options(df_all: pd.DataFrame) -> dict:
    seasons = sorted(df_all["season"].dropna().unique(), reverse=True)
    season = int(seasons[0]) if seasons else 2025

    season_games = df_all[df_all["season"] == season].copy()
    weeks = sorted(season_games["week"].dropna().unique())

    now = pd.Timestamp.now(tz="UTC")
    season_games["_start"] = pd.to_datetime(
        season_games["start_date"], utc=True, errors="coerce"
    )
    past_weeks = sorted(
        season_games.loc[season_games["_start"] <= now, "week"].dropna().unique()
    )
    if past_weeks:
        default_week = int(past_weeks[-1])
    elif weeks:
        default_week = int(weeks[0])
    else:
        default_week = 1

    return {
        "seasons": [int(value) for value in seasons],
        "season": season,
        "weeks": [int(value) for value in weeks],
        "default_week": default_week,
    }


def _parlay_block(df_week: pd.DataFrame, market_snapshots: pd.DataFrame) -> dict:
    """Reproduce the ParlayAPI shadow-quote expander."""
    if not market_snapshots.empty and "source" in market_snapshots.columns:
        parlay = market_snapshots[market_snapshots["source"].eq("parlay_api")].copy()
    else:
        parlay = pd.DataFrame()

    if not parlay.empty:
        week_ids = pd.to_numeric(df_week["game_id"], errors="coerce")
        parlay_week = parlay[
            pd.to_numeric(parlay["game_id"], errors="coerce").isin(week_ids)
        ].copy()
    else:
        parlay_week = pd.DataFrame()

    if parlay_week.empty:
        return {
            "expander": None,
            "caption": "ParlayAPI shadow quotes: no locally captured prices for this week.",
        }

    game_labels = df_week[["game_id", "away_team", "home_team"]].copy()
    game_labels["game_id"] = pd.to_numeric(game_labels["game_id"], errors="coerce")
    parlay_week["game_id"] = pd.to_numeric(parlay_week["game_id"], errors="coerce")
    parlay_week = parlay_week.merge(
        game_labels, on="game_id", how="left", validate="many_to_one"
    )
    parlay_week = (
        parlay_week.sort_values("captured_at")
        .drop_duplicates(["game_id", "sportsbook", "market", "side"], keep="last")
    )
    parlay_week["Game"] = (
        parlay_week["away_team"].astype(str) + " @ " + parlay_week["home_team"].astype(str)
    )
    parlay_week["Market"] = parlay_week["market"].astype(str).str.title()
    parlay_week["Side"] = parlay_week["side"].astype(str).str.title()
    parlay_week["Line"] = pd.to_numeric(parlay_week["line"], errors="coerce").map(
        lambda value: f"{value:+.1f}" if pd.notna(value) else "—"
    )
    parlay_week["American Odds"] = pd.to_numeric(
        parlay_week["odds"], errors="coerce"
    ).map(lambda value: f"{int(value):+d}" if pd.notna(value) else "—")
    parlay_week["Captured UTC"] = parlay_week["captured_at"].dt.strftime("%Y-%m-%d %H:%M")
    if "provider_observed_at" in parlay_week.columns:
        provider_time = pd.to_datetime(
            parlay_week["provider_observed_at"], utc=True, errors="coerce"
        )
        parlay_week["Provider UTC"] = provider_time.dt.strftime("%Y-%m-%d %H:%M")
        parlay_week["Provider UTC"] = parlay_week["Provider UTC"].fillna("—")
    else:
        parlay_week["Provider UTC"] = "—"
    if "is_live" in parlay_week.columns:
        parlay_week["Live"] = parlay_week["is_live"].fillna(False).map(
            lambda value: "Yes" if bool(value) else "No"
        )
    else:
        parlay_week["Live"] = "No"
    if "stale_seconds" in parlay_week.columns:
        parlay_week["Stale Seconds"] = pd.to_numeric(
            parlay_week["stale_seconds"], errors="coerce"
        ).map(lambda value: f"{value:.0f}" if pd.notna(value) else "—")
    else:
        parlay_week["Stale Seconds"] = "—"

    latest_capture = parlay_week["captured_at"].max().strftime("%Y-%m-%d %H:%M UTC")
    table = (
        parlay_week[
            [
                "Game", "sportsbook", "Market", "Side", "Line",
                "American Odds", "Captured UTC", "Provider UTC", "Live", "Stale Seconds",
            ]
        ]
        .rename(columns={"sportsbook": "Book"})
        .sort_values(["Game", "Book", "Market", "Side"])
        .reset_index(drop=True)
    )
    return {
        "caption": None,
        "expander": {
            "label": f"ParlayAPI shadow quotes · {len(parlay_week):,} latest book/market prices",
            "expanded": False,
            "caption": (
                f"Latest local capture: {latest_capture}. These prices are informational only "
                "and do not drive the model or recommendations. Cards use the retained "
                "market snapshot union as a display-only fallback when model-facing lines "
                "are unavailable."
            ),
            "table": {
                "columns": [str(column) for column in table.columns],
                "rows": records(table),
            },
        },
    }


def _compact_rows(
    df_week: pd.DataFrame,
    display_consensus: pd.DataFrame,
    browser_tz,
    kickoff_column: str,
) -> list[dict]:
    """Build the compact-view rows.

    ``DataFrame.iterrows()`` builds a ``Series`` per row and ``Series.get`` is a
    dict lookup on top of that, which made this loop the single most expensive
    part of the endpoint (~200-500 ms for a full week).  Pulling each column out
    as a numpy array once and indexing by position removes both costs; the
    formatting and recommendation logic is unchanged.
    """
    # Vectorised consensus lookup: one dict build instead of one .loc per row.
    consensus: dict[int, tuple[float, float, float, float]] = {}
    if not display_consensus.empty:
        sub = display_consensus.reindex(
            columns=["market_spread", "market_total", "home_moneyline", "away_moneyline"]
        )
        consensus = {
            int(game_id): (spread, total, home_ml, away_ml)
            for game_id, spread, total, home_ml, away_ml in zip(
                display_consensus.index,
                sub["market_spread"].to_numpy(),
                sub["market_total"].to_numpy(),
                sub["home_moneyline"].to_numpy(),
                sub["away_moneyline"].to_numpy(),
            )
        }

    def column(name):
        return df_week[name].to_numpy() if name in df_week.columns else None

    home_teams = column("home_team")
    away_teams = column("away_team")
    win_probs = column("win_prob")
    model_spreads = column("predicted_spread")
    book_spreads = column("market_spread")
    model_totals = column("predicted_total")
    book_totals = column("market_total")
    home_moneylines = column("home_moneyline")
    away_moneylines = column("away_moneyline")
    game_ids = column("game_id")
    home_scores = column("home_score")
    away_scores = column("away_score")

    starts = pd.to_datetime(
        df_week["start_date"] if "start_date" in df_week.columns else pd.Series(dtype="datetime64[ns]"),
        utc=True,
        errors="coerce",
    )
    if len(starts) == len(df_week):
        starts = starts.dt.tz_convert(browser_tz)
        start_list = list(starts)
    else:
        start_list = [pd.NaT] * len(df_week)

    nan = float("nan")
    rows: list[dict] = []
    for i in range(len(df_week)):
        home = home_teams[i] if home_teams is not None else "—"
        away = away_teams[i] if away_teams is not None else "—"
        wp = win_probs[i] if win_probs is not None else nan
        ms = model_spreads[i] if model_spreads is not None else nan
        bs = book_spreads[i] if book_spreads is not None else nan
        mt = model_totals[i] if model_totals is not None else nan
        bt = book_totals[i] if book_totals is not None else nan
        hml = home_moneylines[i] if home_moneylines is not None else nan
        aml = away_moneylines[i] if away_moneylines is not None else nan

        gid = game_ids[i] if game_ids is not None else None
        try:
            fallback = consensus.get(int(gid)) if gid is not None and pd.notna(gid) else None
        except (TypeError, ValueError):
            fallback = None
        if fallback is not None:
            if pd.isna(bs):
                bs = fallback[0]
            if pd.isna(bt):
                bt = fallback[1]
            if pd.isna(hml):
                hml = fallback[2]
            if pd.isna(aml):
                aml = fallback[3]

        spread_rec = (
            generate_spread_pick(home, away, ms, bs)
            if pd.notna(ms) and pd.notna(bs)
            else None
        )
        total_rec = (
            generate_total_pick(home, away, mt, bt)
            if pd.notna(mt) and pd.notna(bt)
            else None
        )
        ml_rec = (
            generate_moneyline_pick(home, away, float(wp), float(hml), float(aml))
            if pd.notna(hml) and pd.notna(aml) and pd.notna(wp)
            else None
        )

        kickoff = start_list[i]
        kickoff_text = (
            f"{kickoff.strftime('%b')} {kickoff.day} · {kickoff.strftime('%I:%M %p')}"
            if pd.notna(kickoff)
            else "—"
        )
        hs = home_scores[i] if home_scores is not None else None
        as_ = away_scores[i] if away_scores is not None else None
        result = f"{int(hs)}–{int(as_)}" if pd.notna(hs) and pd.notna(as_) else "—"

        rows.append(
            {
                "Game": f"{away} @ {home}",
                kickoff_column: kickoff_text,
                "Home win": f"{wp:.0%}" if pd.notna(wp) else "—",
                "Model margin": f"{ms:+.1f}" if pd.notna(ms) else "—",
                "Book spread": f"{bs:+.1f}" if pd.notna(bs) else "—",
                "Spread edge": (
                    f"{spread_rec.edge:.1f} {CONFIDENCE_EMOJI[spread_rec.confidence]}"
                    if spread_rec
                    else "—"
                ),
                "Spread pick": (
                    spread_rec.pick
                    if spread_rec and spread_rec.confidence != Confidence.NONE
                    else "No edge" if spread_rec else "—"
                ),
                "Model O/U": f"{mt:.1f}" if pd.notna(mt) else "—",
                "Book O/U": f"{bt:.1f}" if pd.notna(bt) else "—",
                "O/U edge": (
                    f"{total_rec.edge:.1f} {CONFIDENCE_EMOJI[total_rec.confidence]}"
                    if total_rec
                    else "—"
                ),
                "O/U pick": total_rec.pick if total_rec else "—",
                "Home ML": f"{int(hml):+d}" if pd.notna(hml) else "—",
                "Away ML": f"{int(aml):+d}" if pd.notna(aml) else "—",
                "ML edge": f"{ml_rec.edge:.1%}" if ml_rec else "—",
                "ML pick": ml_rec.pick if ml_rec else "—",
                "Result": result,
            }
        )
    return rows


def build_weekly(
    season: int | None = None,
    week: int | None = None,
    conference: str = "All",
    min_edge: float = 0.0,
    sort_by: str = "Edge (High→Low)",
    timezone_name: str | None = None,
) -> dict:
    """Return the Weekly Predictions payload."""
    df_all = load_feature_matrix()
    if df_all.empty:
        return {
            "page": "weekly",
            "title": "📊 Weekly Predictions",
            "warnings": ["No prediction data is currently published."],
            "stopped": True,
        }

    options = _options(df_all)
    season = options["season"] if season is None else int(season)
    if season not in options["seasons"]:
        season = options["season"]
    week = options["default_week"] if week is None else int(week)
    if week not in options["weeks"]:
        week = options["default_week"]

    release = load_metrics().get("release_decision", {})
    warnings = []
    if release.get("decision") != "promote":
        warnings.append(
            "Research mode: model release gates have not approved betting use."
        )

    df_week = predictions_for(df_all, season, week).copy()

    infos: list[str] = []
    if not models_trained():
        infos.append("Model predictions are not currently published.")
        for column in ["win_prob", "predicted_spread", "predicted_total"]:
            df_week[column] = float("nan")

    conferences = ["All"] + sorted(
        df_week["home_conference"].dropna().unique().tolist()
    )

    if conference != "All":
        df_week = df_week[
            (df_week["home_conference"] == conference)
            | (df_week["away_conference"] == conference)
        ]

    edge_columns: list[str] = []
    if "predicted_spread" in df_week.columns and "market_spread" in df_week.columns:
        df_week["spread_edge"] = (
            df_week["predicted_spread"] + df_week["market_spread"]
        ).abs()
        edge_columns.append("spread_edge")
    if "predicted_total" in df_week.columns and "market_total" in df_week.columns:
        df_week["total_edge"] = (
            df_week["predicted_total"] - df_week["market_total"]
        ).abs()
        edge_columns.append("total_edge")

    if edge_columns:
        df_week["edge"] = df_week[edge_columns].max(axis=1, skipna=True)
        if min_edge > 0:
            df_week = df_week[df_week["edge"] >= min_edge]
        if sort_by == "Edge (High→Low)":
            df_week = df_week.sort_values("edge", ascending=False)
        elif sort_by == "Win Prob":
            df_week = df_week.sort_values("win_prob", ascending=False)

    game_count_line = f"**{len(df_week)} games** — Season {season} · Week {int(week)}"
    captions: list[str] = []
    scopes = set(df_week.get("prediction_scope", pd.Series(dtype=str)).dropna())
    if "walk_forward_oos" in scopes:
        captions.append(
            "Completed games use predictions made by season walk-forward backtests; "
            "unplayed games use the current full-history model."
        )

    if df_week.empty:
        return {
            "page": "weekly",
            "title": "📊 Weekly Predictions",
            "warnings": warnings,
            "infos": infos,
            "captions": captions,
            "options": options,
            "selection": {
                "season": season,
                "week": week,
                "conference": conference,
                "min_edge": min_edge,
                "sort_by": sort_by,
                "conferences": conferences,
                "season_caption": f"Season {int(season)} = the {int(season)} fall schedule · Week {int(week)}",
            },
            "game_count_line": game_count_line,
            "empty_info": "No games match the current filters.",
            "footer": True,
        }

    market_snapshots = load_market_snapshots()
    parlay = _parlay_block(df_week, market_snapshots)

    display_consensus = season_consensus(df_all, season)

    browser_tz, browser_tz_name = resolve_timezone(timezone_name)
    kickoff_column = f"Kickoff ({browser_tz_name})"

    compact_rows = _compact_rows(df_week, display_consensus, browser_tz, kickoff_column)
    table_columns = list(compact_rows[0].keys()) if compact_rows else []
    table_rows = [jsonable(row) for row in compact_rows]

    return {
        "page": "weekly",
        "title": "📊 Weekly Predictions",
        "warnings": warnings,
        "infos": infos,
        "captions": captions,
        "options": options,
        "selection": {
            "season": season,
            "week": week,
            "conference": conference,
            "min_edge": min_edge,
            "sort_by": sort_by,
            "conferences": conferences,
            "season_caption": f"Season {int(season)} = the {int(season)} fall schedule · Week {int(week)}",
        },
        "game_count_line": game_count_line,
        "parlay": parlay,
        "table_caption": (
            f"Compact view — kickoff times shown in {browser_tz_name}. "
            "Use the table’s horizontal scroll to see every market and recommendation."
        ),
        "table": {"columns": table_columns, "rows": table_rows},
        "table_height": min(640, max(140, 36 + 35 * len(compact_rows))),
        "footer": True,
    }
