"""Normalize and append timestamped CFBD sportsbook snapshots."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils.contracts import ensure_utc, validate_line_snapshots
from utils.market import remove_vig
from utils.storage import atomic_write_parquet


def _provider_name(value) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("title") or value.get("id") or "unknown")
    return str(value or "unknown")


def _number(value) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else np.nan


def normalize_cfbd_line_snapshots(
    games: Iterable[dict],
    *,
    captured_at: str | pd.Timestamp,
    ingestion_run_id: str | None = None,
    raw_payload_path: str | None = None,
) -> pd.DataFrame:
    """Convert CFBD's nested provider payload to canonical long market rows."""
    captured = pd.Timestamp(captured_at)
    captured = captured.tz_localize("UTC") if captured.tzinfo is None else captured.tz_convert("UTC")
    rows: list[dict] = []
    for game in games:
        game_id = game.get("id") or game.get("gameId") or game.get("game_id")
        if game_id is None:
            continue
        for quote in game.get("lines") or []:
            sportsbook = _provider_name(quote.get("provider"))
            home_spread = _number(quote.get("spread"))
            total = _number(quote.get("overUnder") or quote.get("over_under"))
            markets = (
                ("spread", "home", home_spread, quote.get("homeSpreadOdds") or quote.get("home_spread_odds")),
                ("spread", "away", -home_spread, quote.get("awaySpreadOdds") or quote.get("away_spread_odds")),
                ("total", "over", total, quote.get("overOdds") or quote.get("over_odds")),
                ("total", "under", total, quote.get("underOdds") or quote.get("under_odds")),
                ("moneyline", "home", np.nan, quote.get("homeMoneyline") or quote.get("home_moneyline")),
                ("moneyline", "away", np.nan, quote.get("awayMoneyline") or quote.get("away_moneyline")),
            )
            for market, side, line, odds in markets:
                price = _number(odds)
                if pd.isna(line) and pd.isna(price):
                    continue
                rows.append(
                    {
                        "game_id": game_id,
                        "sportsbook": sportsbook,
                        "market": market,
                        "side": side,
                        "captured_at": captured,
                        "line": line,
                        "odds": price,
                        "source": "cfbd",
                        "ingestion_run_id": ingestion_run_id,
                        "raw_payload_path": raw_payload_path,
                    }
                )
    columns = [
        "game_id", "sportsbook", "market", "side", "captured_at", "line", "odds", "source",
        "ingestion_run_id", "raw_payload_path",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    result = result.drop_duplicates(
        ["game_id", "sportsbook", "market", "side", "captured_at"], keep="last"
    )
    report = validate_line_snapshots(result)
    report.raise_for_errors()
    return result.reset_index(drop=True)


def build_market_consensus(games: Iterable[dict]) -> pd.DataFrame:
    """Build one robust current/open consensus row per game.

    Lines are medians across providers. Moneyline probabilities are de-vigged
    within each provider before taking the cross-book median; American prices
    themselves must never be averaged to estimate probability.
    """
    rows: list[dict] = []
    for game in games:
        game_id = game.get("id") or game.get("gameId") or game.get("game_id")
        if game_id is None:
            continue
        for quote in game.get("lines") or []:
            home_moneyline = _number(
                quote.get("homeMoneyline") or quote.get("home_moneyline")
            )
            away_moneyline = _number(
                quote.get("awayMoneyline") or quote.get("away_moneyline")
            )
            market_home_prob = np.nan
            if (
                np.isfinite(home_moneyline) and np.isfinite(away_moneyline)
                and home_moneyline != 0 and away_moneyline != 0
            ):
                market_home_prob = float(
                    remove_vig([home_moneyline, away_moneyline])[0]
                )
            rows.append(
                {
                    "game_id": game_id,
                    "season": game.get("season") or game.get("year"),
                    "provider": _provider_name(quote.get("provider")),
                    "market_spread": _number(quote.get("spread")),
                    "market_spread_open": _number(
                        quote.get("spreadOpen") or quote.get("spread_open")
                    ),
                    "market_total": _number(
                        quote.get("overUnder") or quote.get("over_under")
                    ),
                    "market_total_open": _number(
                        quote.get("overUnderOpen") or quote.get("over_under_open")
                    ),
                    "home_moneyline": home_moneyline,
                    "away_moneyline": away_moneyline,
                    "market_home_prob": market_home_prob,
                }
            )
    quotes = pd.DataFrame(rows)
    if quotes.empty:
        return pd.DataFrame()

    # Historical raw JSON may preserve numeric CFBD identifiers as strings.
    # Normalize them before grouping/upserting so PyArrow never receives a
    # mixed integer/string ``game_id`` column.
    quotes["game_id"] = pd.to_numeric(quotes["game_id"], errors="raise").astype("int64")

    numeric = [
        "market_spread", "market_spread_open", "market_total",
        "market_total_open", "home_moneyline", "away_moneyline",
        "market_home_prob",
    ]
    for column in numeric:
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    grouped = quotes.groupby("game_id", sort=False)
    consensus = grouped.agg(
        season=("season", "max"),
        market_spread=("market_spread", "median"),
        market_spread_open=("market_spread_open", "median"),
        market_spread_dispersion=("market_spread", "std"),
        market_spread_book_count=("market_spread", "count"),
        market_total=("market_total", "median"),
        market_total_open=("market_total_open", "median"),
        market_total_dispersion=("market_total", "std"),
        market_total_book_count=("market_total", "count"),
        home_moneyline=("home_moneyline", "median"),
        away_moneyline=("away_moneyline", "median"),
        market_home_prob=("market_home_prob", "median"),
        moneyline_book_count=("market_home_prob", "count"),
    ).reset_index()
    for column in ("market_spread_dispersion", "market_total_dispersion"):
        consensus[column] = consensus[column].fillna(0.0)
    consensus["market_spread_move"] = (
        consensus["market_spread"] - consensus["market_spread_open"]
    )
    consensus["market_total_move"] = (
        consensus["market_total"] - consensus["market_total_open"]
    )
    return consensus


def append_line_snapshots(snapshots: pd.DataFrame, path: str | Path) -> Path:
    """Append idempotently and replace the compressed Parquet artifact atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if snapshots.empty:
        return destination
    existing = pd.read_parquet(destination) if destination.exists() else pd.DataFrame()
    combined = pd.concat([existing, snapshots], ignore_index=True)
    combined["captured_at"] = ensure_utc(combined["captured_at"])
    keys = ["game_id", "sportsbook", "market", "side", "captured_at"]
    combined = combined.sort_values("captured_at").drop_duplicates(keys, keep="last")
    validate_line_snapshots(combined).raise_for_errors()
    return atomic_write_parquet(combined, destination)
