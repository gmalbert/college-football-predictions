"""Batch NCAAF main-market adapter for OddsAPI.io."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import islice
from math import isfinite
from typing import Iterable

import pandas as pd
import requests

from utils.config import get_secret
from utils.logger import get_logger
from utils.odds_api import match_market_event

logger = get_logger(__name__)

BASE_URL = "https://api.odds-api.io/v3"
TIMEOUT = 20


def _api_key() -> str:
    key = get_secret("odds_api_io", "key").strip()
    if not key:
        raise ValueError("ODDS_API_IO_KEY is empty.")
    return key


def _bookmakers() -> str:
    books = get_secret("odds_api_io", "bookmakers").strip()
    if not books:
        raise ValueError("ODDS_API_IO_BOOKMAKERS is empty.")
    return books


def is_configured() -> bool:
    try:
        return bool(_api_key() and _bookmakers())
    except ValueError:
        return False


def _chunks(values: list[object], size: int) -> Iterable[list[object]]:
    iterator = iter(values)
    while chunk := list(islice(iterator, size)):
        yield chunk


def get_ncaaf_odds() -> list[dict]:
    """Fetch pending NCAAF events and their selected-book main markets.

    OddsAPI.io exposes NCAAF under its ``american-football`` sport catalogue. Its multi-odds
    endpoint accepts up to ten events per request, keeping a full board well
    inside the free tier's daily request allowance.
    """
    try:
        key, books = _api_key(), _bookmakers()
        now = datetime.now(timezone.utc)
        events_response = requests.get(
            f"{BASE_URL}/events",
            params={
                "apiKey": key,
                "sport": "american-football",
                "status": "pending",
                "from": now.isoformat().replace("+00:00", "Z"),
                "to": (now + timedelta(days=14)).isoformat().replace("+00:00", "Z"),
            },
            timeout=TIMEOUT,
        )
        events_response.raise_for_status()
        deadline = now + timedelta(days=14)
        events = []
        for event in events_response.json():
            league = event.get("league", {})
            event_time = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
            is_ncaaf = (
                "college" in str(league.get("name", "")).casefold()
                or "ncaaf" in str(league.get("slug", "")).casefold()
            )
            if is_ncaaf and pd.notna(event_time) and now <= event_time.to_pydatetime() <= deadline:
                events.append(event)
        results: list[dict] = []
        for event_ids in _chunks([event["id"] for event in events], 10):
            response = requests.get(
                f"{BASE_URL}/odds/multi",
                params={
                    "apiKey": key, "eventIds": ",".join(map(str, event_ids)), "bookmakers": books,
                    "markets": "ML,Spread,Totals",
                },
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            results.extend(response.json())
        logger.info("OddsAPI.io NCAAF snapshot received: %s events, %s requests", len(results), 1 + (len(events) + 9) // 10)
        return results
    except requests.HTTPError as exc:
        # ``requests`` includes the entire query string (and its API key) in
        # an exception's string representation. Never log that representation.
        logger.error("OddsAPI.io NCAAF snapshot HTTP error: %s", exc.response.status_code)
        return []
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.error("OddsAPI.io NCAAF snapshot error: %s", type(exc).__name__)
        return []


def _american(decimal: object) -> int | None:
    try:
        value = float(decimal)
    except (TypeError, ValueError):
        return None
    if not isfinite(value) or value <= 1:
        return None
    return round((value - 1) * 100) if value >= 2 else round(-100 / (value - 1))


def _main_line(odds: list[dict], sides: tuple[str, str], *, total: bool = False) -> tuple[float, int, int] | None:
    """Choose a complete full-game line, preferring the provider's main marker."""
    best = None
    best_rank = None
    for row in odds:
        try:
            raw_line = row.get("hdp")
            if total and raw_line is None:
                raw_line = row.get("max")  # Retained payloads used this key.
            line = float(raw_line)
            left, right = (float(row[side]) for side in sides)
        except (KeyError, TypeError, ValueError):
            continue
        if not all(isfinite(value) for value in (line, left, right)) or min(left, right) <= 1:
            continue
        # For an unlabeled ladder, use the closest two-sided price to an even
        # market. The first displayed line can be a deeply juiced alternate.
        rank = (row.get("main") is not True, abs(left - right) / (left + right))
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = (line, _american(left), _american(right))
    return best


def to_cfbd_line_payload(events: Iterable[dict], games: pd.DataFrame, *, season: int) -> list[dict]:
    """Map selected OddsAPI.io book quotes to the project's canonical payload."""
    schedule = games[games["season"].eq(season)].copy() if not games.empty else games
    rows: list[dict] = []
    for event in events:
        home, away = event.get("home"), event.get("away")
        match = match_market_event(schedule, home, away, event.get("date"))
        if match is None:
            logger.warning("Skipping unmatched/ambiguous OddsAPI.io event: %s at %s", away, home)
            continue
        game, reversed_teams = match
        team_keys = ("away", "home") if reversed_teams else ("home", "away")
        lines: list[dict] = []
        for book, markets in (event.get("bookmakers") or {}).items():
            quote = {"provider": book}
            for market in markets or []:
                name = str(market.get("name", "")).strip().casefold()
                odds = market.get("odds") or []
                if name in {"ml", "moneyline"}:
                    if odds:
                        quote["homeMoneyline"] = _american(odds[0].get(team_keys[0]))
                        quote["awayMoneyline"] = _american(odds[0].get(team_keys[1]))
                elif name in {"spread", "spreads", "handicap"}:
                    main = _main_line(odds, team_keys)
                    if main is not None:
                        line, quote["homeSpreadOdds"], quote["awaySpreadOdds"] = main
                        quote["spread"] = -line if reversed_teams else line
                elif name in {"total", "totals", "over/under"}:
                    main = _main_line(odds, ("over", "under"), total=True)
                    if main is not None:
                        quote["overUnder"], quote["overOdds"], quote["underOdds"] = main
            if len(quote) > 1:
                lines.append(quote)
        if lines:
            rows.append({"id": game["game_id"], "season": season, "lines": lines})
    return rows
