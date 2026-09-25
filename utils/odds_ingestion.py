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
    source: str = "cfbd",
) -> pd.DataFrame:
    """Convert CFBD's nested provider payload to canonical long market rows."""
    captured = pd.Timestamp(captured_at)
    captured = captured.tz_localize("UTC") if captured.tzinfo is None else captured.tz_convert("UTC")
    rows: list[dict] = []
    for game in games:
        game_id = game.get("id") or game.get("gameId") or game.get("game_id")
        if game_id is None:
            continue
        row_source = str(game.get("source") or source)
        available_at = pd.to_datetime(game.get("available_at"), utc=True, errors="coerce")
        if pd.isna(available_at):
            available_at = captured
        provider_event_id = game.get("provider_event_id")
        provider_observed_at = pd.to_datetime(
            game.get("provider_observed_at"), utc=True, errors="coerce"
        )
        if pd.isna(provider_observed_at):
            provider_observed_at = pd.NaT
        game_is_live = game.get("is_live")
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
                        "available_at": available_at,
                        "provider_event_id": quote.get("provider_event_id") or provider_event_id,
                        "provider_observed_at": pd.to_datetime(
                            quote.get("provider_observed_at"), utc=True, errors="coerce"
                        ) if quote.get("provider_observed_at") else provider_observed_at,
                        "is_live": quote.get("is_live", game_is_live),
                        "stale_seconds": quote.get("stale_seconds"),
                        "topped_up": quote.get("topped_up"),
                        "line": line,
                        "odds": price,
                        "source": row_source,
                        "ingestion_run_id": ingestion_run_id,
                        "raw_payload_path": raw_payload_path,
                    }
                )
    columns = [
        "game_id", "sportsbook", "market", "side", "captured_at", "available_at",
        "provider_event_id", "provider_observed_at", "is_live", "stale_seconds", "topped_up",
        "line", "odds", "source",
        "ingestion_run_id", "raw_payload_path",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    result = result.drop_duplicates(
        ["game_id", "source", "sportsbook", "market", "side", "captured_at"], keep="last"
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


def build_market_consensus_from_snapshots(
    snapshots: pd.DataFrame,
    games: pd.DataFrame,
    *,
    season: int,
    exclude_sources: Iterable[str] = ("parlay_api",),
) -> pd.DataFrame:
    """Build current market features from the retained snapshot union.

    ``lines.parquet`` historically came from one provider's nested response.
    That made a later provider refresh erase games that were present in an
    earlier feed.  The immutable long-form snapshot table is the durable
    source of truth, so use its latest quote per book/market/side and combine
    all non-shadow sources into one consensus for the requested season.

    ParlayAPI is excluded by default because it remains a shadow source and
    must not change production model inputs.  The Streamlit page can still
    use those rows as a display-only fallback.
    """
    required = {"game_id", "sportsbook", "market", "side", "captured_at", "line", "odds"}
    missing = sorted(required.difference(snapshots.columns))
    if missing or games.empty:
        return pd.DataFrame()

    schedule = games[games["season"].eq(season)].copy()
    if schedule.empty:
        return pd.DataFrame()
    game_ids = pd.to_numeric(schedule["game_id"], errors="coerce").dropna().astype("int64")

    frame = snapshots.copy()
    frame["game_id"] = pd.to_numeric(frame["game_id"], errors="coerce")
    # Boolean masking already returns a new frame, and pandas copies on write, so
    # the explicit .copy() calls that used to follow only forced extra full
    # materialisations of a 200k-row frame.
    frame = frame[frame["game_id"].isin(set(game_ids))]
    excluded = {str(source) for source in exclude_sources}
    if "source" in frame.columns and excluded:
        frame = frame[~frame["source"].astype(str).isin(excluded)]
    if frame.empty:
        return pd.DataFrame()

    frame["captured_at"] = ensure_utc(frame["captured_at"])
    frame["line"] = pd.to_numeric(frame["line"], errors="coerce")
    frame["odds"] = pd.to_numeric(frame["odds"], errors="coerce")
    frame = frame.dropna(subset=["game_id", "captured_at"])
    if frame.empty:
        return pd.DataFrame()
    frame["game_id"] = frame["game_id"].astype("int64")

    quote_keys = ["game_id", "sportsbook", "market", "side"]
    ordered = frame.sort_values("captured_at")
    current = ordered.drop_duplicates(quote_keys, keep="last")
    opening = ordered.drop_duplicates(quote_keys, keep="first")

    # Aggregate once per (game, market, side) instead of masking a frame for
    # every game and every book.  ``market`` and ``side`` are Arrow-backed
    # strings, so each ``.eq()`` boxes a Python object per row; the previous
    # ~9,250 masks cost ~15s of a ~19s call, and the count grows with the
    # retained snapshot history (200k quotes by late September).
    line_key = ["game_id", "market", "side"]
    current_stats = (
        current.groupby(line_key, sort=False)["line"]
        .agg(["median", "std", "count"])
        .to_dict("index")
    )
    opening_stats = (
        opening.groupby(line_key, sort=False)["line"]
        .agg(["median", "std", "count"])
        .to_dict("index")
    )

    # A book only contributes to the moneyline consensus when it quoted one, so
    # index the moneyline rows rather than every book in the game.
    book_moneyline = (
        current.loc[current["market"].eq("moneyline")]
        .groupby(["game_id", "sportsbook", "side"], sort=False)["odds"]
        .median()
        .to_dict()
    )

    def line_stat(stats: dict, game_id: int, market: str, side: str) -> dict | None:
        return stats.get((game_id, market, side))

    def line_median(stats: dict, game_id: int, market: str, side: str) -> float:
        entry = line_stat(stats, game_id, market, side)
        return float(entry["median"]) if entry is not None else np.nan

    def _count_of(entry: dict | None) -> int:
        return int(entry["count"]) if entry is not None else 0

    def _std_of(entry: dict | None) -> float:
        count = _count_of(entry)
        return float(entry["std"]) if count > 1 else 0.0

    rows: list[dict] = []
    for game_id, group in current.groupby("game_id", sort=False):
        home_spread = line_median(current_stats, game_id, "spread", "home")
        open_spread = line_median(opening_stats, game_id, "spread", "home")
        total = line_median(current_stats, game_id, "total", "over")
        if pd.isna(total):
            total = line_median(current_stats, game_id, "total", "under")
        open_total = line_median(opening_stats, game_id, "total", "over")
        if pd.isna(open_total):
            open_total = line_median(opening_stats, game_id, "total", "under")

        spreads = line_stat(current_stats, game_id, "spread", "home")
        totals = line_stat(current_stats, game_id, "total", "over")
        if _count_of(totals) == 0:
            totals = line_stat(current_stats, game_id, "total", "under")

        home_moneylines: list[float] = []
        away_moneylines: list[float] = []
        market_home_probs: list[float] = []
        for book in group["sportsbook"].unique():
            home_ml = book_moneyline.get((game_id, book, "home"))
            away_ml = book_moneyline.get((game_id, book, "away"))
            home_priced = home_ml is not None and pd.notna(home_ml)
            away_priced = away_ml is not None and pd.notna(away_ml)
            if home_priced:
                home_moneylines.append(float(home_ml))
            if away_priced:
                away_moneylines.append(float(away_ml))
            if home_priced and away_priced and home_ml != 0 and away_ml != 0:
                market_home_probs.append(float(remove_vig([home_ml, away_ml])[0]))

        rows.append(
            {
                "game_id": int(game_id),
                "season": int(season),
                "market_spread": home_spread,
                "market_spread_open": open_spread,
                "market_spread_dispersion": _std_of(spreads),
                "market_spread_book_count": _count_of(spreads),
                "market_total": total,
                "market_total_open": open_total,
                "market_total_dispersion": _std_of(totals),
                "market_total_book_count": _count_of(totals),
                "home_moneyline": float(np.median(home_moneylines)) if home_moneylines else np.nan,
                "away_moneyline": float(np.median(away_moneylines)) if away_moneylines else np.nan,
                "market_home_prob": float(np.median(market_home_probs)) if market_home_probs else np.nan,
                "moneyline_book_count": len(market_home_probs),
            }
        )

    consensus = pd.DataFrame(rows)
    if consensus.empty:
        return consensus
    consensus["market_spread_move"] = consensus["market_spread"] - consensus["market_spread_open"]
    consensus["market_total_move"] = consensus["market_total"] - consensus["market_total_open"]
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
    if "source" not in combined.columns:
        combined["source"] = "cfbd"
    combined["source"] = combined["source"].fillna("cfbd")
    if "available_at" not in combined.columns:
        combined["available_at"] = combined["captured_at"]
    else:
        combined["available_at"] = ensure_utc(combined["available_at"])
        combined["available_at"] = combined["available_at"].fillna(combined["captured_at"])
    keys = ["game_id", "source", "sportsbook", "market", "side", "captured_at"]
    combined = combined.sort_values("captured_at").drop_duplicates(keys, keep="last")
    validate_line_snapshots(combined).raise_for_errors()
    return atomic_write_parquet(combined, destination)
