"""Append-only shadow ledger, quote selection, CLV, and settlement helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils.contracts import ensure_utc
from utils.market import closing_line_value, settle_bet
from utils.storage import DATA_DIR, atomic_write_parquet


LEDGER_PATH = DATA_DIR / "processed" / "bet_ledger.parquet"


LEDGER_COLUMNS = [
    "ledger_id", "game_id", "market", "side", "prediction_time", "kickoff",
    "sportsbook", "taken_line", "taken_odds", "model_probability", "market_probability",
    "expected_value", "stake", "strategy_version", "model_version", "release_id",
    "source_snapshot_at", "ingestion_run_id", "status", "closing_line", "closing_odds",
    "clv", "result", "profit", "settled_at",
]


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
    choices = choices[choices["captured_at"] <= pd.Timestamp(as_of).tz_convert("UTC")]
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
    frame["market"] = "total"
    frame["status"] = "shadow_not_a_bet"
    frame["stake"] = 0.0
    frame["market_probability"] = 0.5
    frame["expected_value"] = np.nan
    frame["closing_line"] = np.nan
    frame["closing_odds"] = np.nan
    frame["clv"] = np.nan
    frame["result"] = None
    frame["profit"] = np.nan
    frame["settled_at"] = pd.NaT
    frame["ledger_id"] = (
        frame["strategy_version"].astype(str) + ":" + frame["game_id"].astype(str)
        + ":" + frame["side"].astype(str) + ":" + frame["prediction_time"].astype(str)
    )
    for column in LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    existing = pd.read_parquet(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame(columns=LEDGER_COLUMNS)
    combined = pd.concat([existing, frame[LEDGER_COLUMNS]], ignore_index=True)
    combined = combined.drop_duplicates("ledger_id", keep="last")
    return atomic_write_parquet(combined, LEDGER_PATH)


def enrich_ledger_with_closing_and_results(
    ledger: pd.DataFrame,
    snapshots: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the last pre-kickoff same-book quote and settle completed shadow rows."""
    if ledger.empty:
        return ledger
    result = ledger.copy()
    result["prediction_time"] = ensure_utc(result["prediction_time"])
    result["kickoff"] = ensure_utc(result["kickoff"])
    quotes = snapshots.copy()
    quotes["captured_at"] = ensure_utc(quotes["captured_at"])
    finals = games[["game_id", "home_score", "away_score"]].drop_duplicates("game_id")
    result = result.merge(finals, on="game_id", how="left", validate="many_to_one")
    for index, row in result.iterrows():
        eligible = quotes[
            (pd.to_numeric(quotes["game_id"], errors="coerce") == int(row["game_id"]))
            & quotes["market"].eq(row["market"])
            & quotes["side"].eq(row["side"])
            & (quotes["captured_at"] <= row["kickoff"])
        ]
        if pd.notna(row.get("sportsbook")):
            same_book = eligible[eligible["sportsbook"].eq(row["sportsbook"])]
            eligible = same_book if not same_book.empty else eligible
        if not eligible.empty:
            closing = eligible.sort_values("captured_at").iloc[-1]
            result.at[index, "closing_line"] = closing.get("line")
            result.at[index, "closing_odds"] = closing.get("odds")
            result.at[index, "clv"] = closing_line_value(
                market=str(row["market"]), taken_line=row.get("taken_line"),
                closing_line=closing.get("line"), taken_odds=row.get("taken_odds"),
                closing_odds=closing.get("odds"),
            )
        if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
            actual_total = float(row["home_score"] + row["away_score"])
            settlement = settle_bet(
                market="total", side=str(row["side"]), line=float(row["taken_line"]),
                odds=float(row["taken_odds"]) if pd.notna(row.get("taken_odds")) else -110.0,
                home_score=float(row["home_score"]), away_score=float(row["away_score"]),
                stake=float(row.get("stake") or 0.0),
            )
            result.at[index, "result"] = settlement.result
            result.at[index, "profit"] = settlement.profit
            result.at[index, "status"] = "settled_shadow"
            result.at[index, "settled_at"] = datetime.now(timezone.utc)
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
