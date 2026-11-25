"""Append-only shadow ledger, quote selection, CLV, and settlement helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.contracts import ensure_utc
from utils.evaluation import cluster_bootstrap_interval
from utils.market import closing_line_value, settle_bet
from utils.odds_ingestion import recorded_provider_matches
from utils.storage import DATA_DIR, atomic_write_json, atomic_write_parquet


LEDGER_PATH = DATA_DIR / "processed" / "bet_ledger.parquet"


LEDGER_COLUMNS = [
    "ledger_id", "game_id", "market", "side", "prediction_time", "kickoff",
    "sportsbook", "taken_line", "taken_odds", "model_probability", "market_probability",
    "expected_value", "stake", "strategy_version", "model_version", "release_id",
    "source_snapshot_at", "odds_source", "ingestion_run_id", "status",
    "closing_snapshot_at", "closing_line", "closing_odds",
    "clv", "result", "profit", "settled_at",
]

CLV_MIN_SIGNALS = 150
CLV_MIN_WEEKS = 8
PROSPECTIVE_STRATEGIES = {
    ("moneyline", None): "moneyline-v2.3-priced",
    ("spread", None): "spread-v2.3-priced",
    ("total", "over"): "total-over-v2.3-priced",
}


def prospective_clv_summary(ledger: pd.DataFrame) -> dict[str, dict]:
    """Summarize settled prospective CLV using week-clustered confidence bounds."""
    result: dict[str, dict] = {}
    timing = {"source_snapshot_at", "prediction_time", "closing_snapshot_at", "kickoff"}
    if not ledger.empty and timing.issubset(ledger.columns):
        entry_quote = pd.to_datetime(ledger["source_snapshot_at"], utc=True, errors="coerce")
        entry = pd.to_datetime(ledger["prediction_time"], utc=True, errors="coerce")
        close = pd.to_datetime(ledger["closing_snapshot_at"], utc=True, errors="coerce")
        kickoff = pd.to_datetime(ledger["kickoff"], utc=True, errors="coerce")
        ledger = ledger[entry_quote.le(entry) & entry.lt(close) & close.lt(kickoff)]
    else:
        ledger = ledger.iloc[:0]
    for (market, side), strategy in PROSPECTIVE_STRATEGIES.items():
        key = f"{market}:{side or 'all'}"
        if ledger.empty:
            subset = pd.DataFrame()
        else:
            mask = (
                ledger["market"].eq(market)
                & ledger["status"].eq("settled_shadow")
                & ledger["strategy_version"].eq(strategy)
            )
            if side is not None:
                mask &= ledger["side"].eq(side)
            subset = ledger[mask].copy()
        if not subset.empty:
            subset["prediction_time"] = ensure_utc(subset["prediction_time"])
            subset = (
                subset.sort_values("prediction_time", kind="stable")
                .drop_duplicates("game_id", keep="first")
            )
            subset["clv"] = pd.to_numeric(subset["clv"], errors="coerce")
            subset = subset[np.isfinite(subset["clv"])]
        n = int(len(subset))
        mean_ci = (None, None)
        positive_ci = (None, None)
        if n:
            kickoff = pd.to_datetime(subset.get("kickoff"), utc=True, errors="coerce")
            subset["season_week"] = kickoff.dt.strftime("%Y-%U").fillna("unknown")
            if subset["season_week"].nunique() >= 2:
                mean_ci = cluster_bootstrap_interval(
                    subset, lambda frame: float(frame["clv"].mean()),
                    cluster="season_week", samples=1000,
                )
                positive_ci = cluster_bootstrap_interval(
                    subset, lambda frame: float((frame["clv"] > 0).mean()),
                    cluster="season_week", samples=1000,
                )
        mean_clv = float(subset["clv"].mean()) if n else None
        positive_rate = float((subset["clv"] > 0).mean()) if n else None
        passed = bool(
            n >= CLV_MIN_SIGNALS
            and subset["season_week"].nunique() >= CLV_MIN_WEEKS
            and mean_ci[0] is not None and mean_ci[0] > 0
            and positive_ci[0] is not None and positive_ci[0] > 0.5
        )
        result[key] = {
            "market": market,
            "side": side,
            "strategy_version": strategy,
            "n": n,
            "distinct_weeks": int(subset["season_week"].nunique()) if n else 0,
            "mean_clv": mean_clv,
            "mean_clv_ci_95": list(mean_ci),
            "positive_clv_rate": positive_rate,
            "positive_clv_rate_ci_95": list(positive_ci),
            "minimum_signals": CLV_MIN_SIGNALS,
            "minimum_weeks": CLV_MIN_WEEKS,
            "passed": passed,
        }
    return result


def refresh_prospective_release_metadata() -> dict:
    """Refresh CLV evidence and fail-closed release decisions in model metrics."""
    from utils.evaluation import evaluate_release_gates
    from utils.model_registry import promotion_decision
    from utils.release import market_release_statuses
    from utils.storage import MODELS_DIR

    metrics_path = MODELS_DIR / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    ledger = pd.read_parquet(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame(columns=LEDGER_COLUMNS)
    metrics["prospective_clv"] = prospective_clv_summary(ledger)
    metrics["release_gates"] = evaluate_release_gates(metrics)
    metrics["release_decision"] = promotion_decision(metrics["release_gates"])
    metrics["market_release_status"] = market_release_statuses(metrics)
    atomic_write_json(metrics_path, metrics)
    return metrics["release_decision"]


def select_quote_as_of(
    snapshots: pd.DataFrame,
    *,
    game_id: int,
    market: str,
    side: str,
    as_of: pd.Timestamp,
) -> pd.Series | None:
    """Return the most recent quote at or before a declared prediction timestamp."""
    if snapshots.empty:
        return None
    choices = snapshots[
        (pd.to_numeric(snapshots["game_id"], errors="coerce") == int(game_id))
        & snapshots["market"].eq(market)
        & snapshots["side"].eq(side)
    ].copy()
    choices["captured_at"] = ensure_utc(choices["captured_at"])
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    choices = choices[
        choices["captured_at"].between(cutoff - pd.Timedelta(hours=24), cutoff)
    ]
    if "is_live" in choices.columns:
        choices = choices[~choices["is_live"].fillna(False).astype(bool)]
    choices["odds"] = pd.to_numeric(choices["odds"], errors="coerce")
    choices = choices[choices["odds"].notna() & choices["odds"].ne(0)]
    if choices.empty:
        return None
    # Prefer a priced quote, then freshest observation; this stays deterministic.
    choices["_priced"] = pd.to_numeric(choices["odds"], errors="coerce").notna()
    return choices.sort_values(["_priced", "captured_at"], ascending=[False, False]).iloc[0]


def append_shadow_signals(signals: list[dict], *, metadata: dict) -> Path:
    """Append idempotent shadow signals to the canonical bet ledger."""
    if not signals:
        return LEDGER_PATH
    frame = pd.DataFrame(signals)
    frame["prediction_time"] = ensure_utc(frame["prediction_time"])
    frame["kickoff"] = ensure_utc(frame["kickoff"])
    for key, value in metadata.items():
        frame[key] = value
    frame["status"] = "shadow_not_a_bet"
    frame["stake"] = 0.0
    frame["closing_snapshot_at"] = pd.NaT
    frame["closing_line"] = np.nan
    frame["closing_odds"] = np.nan
    frame["clv"] = np.nan
    frame["result"] = None
    frame["profit"] = np.nan
    frame["settled_at"] = pd.NaT
    frame["ledger_id"] = (
        frame["strategy_version"].astype(str) + ":" + frame["game_id"].astype(str)
        + ":" + frame["market"].astype(str)
    )
    for column in LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    existing = pd.read_parquet(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame(columns=LEDGER_COLUMNS)
    combined = pd.concat([existing, frame[LEDGER_COLUMNS]], ignore_index=True)
    combined["prediction_time"] = ensure_utc(combined["prediction_time"])
    combined = (
        combined.sort_values("prediction_time", kind="stable")
        .drop_duplicates(
            ["strategy_version", "game_id", "market"], keep="first"
        )
    )
    return atomic_write_parquet(combined, LEDGER_PATH)


def enrich_ledger_with_closing_and_results(
    ledger: pd.DataFrame,
    snapshots: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Grade entries against a later pre-kickoff quote from the same book."""
    required = {"game_id", "market", "side", "sportsbook", "captured_at", "line", "odds"}
    if ledger.empty or not required.issubset(snapshots.columns):
        return ledger

    result = ledger.copy()
    result["prediction_time"] = ensure_utc(result["prediction_time"])
    result["kickoff"] = ensure_utc(result["kickoff"])
    result["settled_at"] = ensure_utc(result["settled_at"]).astype("datetime64[ns, UTC]")
    quotes = snapshots.loc[recorded_provider_matches(snapshots)].copy()
    quotes["game_id"] = pd.to_numeric(quotes["game_id"], errors="coerce")
    quotes["captured_at"] = ensure_utc(quotes["captured_at"])
    quotes["odds"] = pd.to_numeric(quotes["odds"], errors="coerce")
    quotes["line"] = pd.to_numeric(quotes["line"], errors="coerce")
    quotes = quotes[np.isfinite(quotes["odds"]) & quotes["odds"].abs().ge(100)]
    if "is_live" in quotes.columns:
        live = quotes["is_live"].astype("string").str.strip().str.casefold().isin({"1", "true", "yes", "y"})
        quotes = quotes[~live]
    if "source" in quotes.columns:
        quotes["source"] = quotes["source"].astype("string").str.strip().str.casefold()
        quotes = quotes[~quotes["source"].isin({"parlay", "parlay_api"})]
    indexed_quotes = quotes.sort_values("captured_at").groupby(
        ["game_id", "market", "side", "sportsbook"], sort=False
    )
    finals = games[["game_id", "home_score", "away_score"]].drop_duplicates("game_id")
    result = result.merge(finals, on="game_id", how="left", validate="many_to_one")
    result["closing_snapshot_at"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result[["clv", "closing_line", "closing_odds"]] = np.nan
    settled_at = datetime.now(timezone.utc)
    for index, row in result.iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        market = str(row["market"])
        taken_odds = row.get("taken_odds")
        taken_line = row.get("taken_line")
        if pd.isna(taken_odds) or (market != "moneyline" and pd.isna(taken_line)):
            continue
        settlement = settle_bet(
            market=market, side=str(row["side"]), line=taken_line, odds=float(taken_odds),
            home_score=float(row["home_score"]), away_score=float(row["away_score"]),
            stake=0.0,
        )
        result.at[index, "result"] = settlement.result
        result.at[index, "profit"] = settlement.profit
        result.at[index, "status"] = "settled_shadow"
        if pd.isna(row.get("settled_at")):
            result.at[index, "settled_at"] = settled_at

        key = (row["game_id"], market, row["side"], row["sportsbook"])
        if key not in indexed_quotes.groups:
            continue
        eligible = indexed_quotes.get_group(key)
        eligible = eligible[
            eligible["captured_at"].gt(row["prediction_time"])
            & eligible["captured_at"].lt(row["kickoff"])
        ]
        if pd.notna(row.get("odds_source")):
            if "source" not in eligible.columns:
                continue
            eligible = eligible[eligible["source"].eq(row["odds_source"])]
        if market != "moneyline":
            eligible = eligible[np.isfinite(eligible["line"])]
        if eligible.empty:
            continue
        closing = eligible.iloc[-1]
        result.at[index, "closing_snapshot_at"] = closing["captured_at"]
        result.at[index, "closing_line"] = closing["line"]
        result.at[index, "closing_odds"] = closing["odds"]
        result.at[index, "clv"] = closing_line_value(
            market=market, taken_line=taken_line, closing_line=closing["line"],
            taken_odds=taken_odds, closing_odds=closing["odds"], side=str(row["side"]),
        )
    return result.drop(columns=["home_score", "away_score"], errors="ignore")


def settle_shadow_ledger() -> Path:
    """Refresh close/settlement fields for all recorded shadow signals."""
    if not LEDGER_PATH.exists():
        return atomic_write_parquet(pd.DataFrame(columns=LEDGER_COLUMNS), LEDGER_PATH)
    snapshots_path = DATA_DIR / "processed" / "line_snapshots.parquet"
    games_path = DATA_DIR / "processed" / "games.parquet"
    if not snapshots_path.exists() or not games_path.exists():
        return LEDGER_PATH
    result = enrich_ledger_with_closing_and_results(
        pd.read_parquet(LEDGER_PATH), pd.read_parquet(snapshots_path), pd.read_parquet(games_path)
    )
    return atomic_write_parquet(result, LEDGER_PATH)
