"""ParlayAPI adapter for shadow NCAAF market capture.

The provider advertises a The Odds API-compatible event/market shape.  This
module deliberately maps responses into the repository's existing CFBD-shaped
intermediate payload so the normal snapshot writer can retain one canonical
quote grain without changing the production model path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable
import unicodedata

import pandas as pd
import requests

from utils.config import get_secret
from utils.logger import get_logger
from utils.odds_api import match_scheduled_game

logger = get_logger(__name__)

BASE_URL = "https://parlay-api.com/v1/sports/americanfootball_ncaaf/odds"
TIMEOUT = 20
SOURCE = "parlay_api"


def _api_key() -> str:
    """Return the configured ParlayAPI key without ever logging its value."""
    key = get_secret("parlay", "api_key").strip()
    if not key:
        raise ValueError("PARLAY_API_KEY is empty.")
    return key


def is_configured() -> bool:
    """Whether a ParlayAPI key is available."""
    try:
        return bool(_api_key())
    except ValueError:
        return False


def get_ncaaf_odds() -> list[dict]:
    """Fetch current US NCAAF h2h, spread, and total markets."""
    try:
        response = requests.get(
            BASE_URL,
            headers={"X-API-Key": _api_key()},
            params={
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        logger.info(
            "ParlayAPI NCAAF snapshot received; credits used=%s remaining=%s",
            response.headers.get("x-credits-used", response.headers.get("x-requests-used", "unknown")),
            response.headers.get("x-credits-remaining", response.headers.get("x-requests-remaining", "unknown")),
        )
        payload = response.json()
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            events = payload.get("events") or payload.get("data")
            if isinstance(events, list):
                return events
        logger.error("ParlayAPI NCAAF response did not contain an event list")
        return []
    except requests.HTTPError as exc:
        # Do not log ``str(exc)``: some clients include request details in it.
        status = exc.response.status_code if exc.response is not None else "unknown"
        logger.error("ParlayAPI NCAAF HTTP error: %s", status)
        return []
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.error("ParlayAPI NCAAF snapshot error: %s", type(exc).__name__)
        return []


def _team_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return "".join(character for character in normalized.casefold() if character.isalnum())


def _same_team(left: object, right: object) -> bool:
    left_key, right_key = _team_key(left), _team_key(right)
    return bool(left_key and right_key) and (
        left_key == right_key
        or (len(left_key) >= 3 and right_key.startswith(left_key))
        or (len(right_key) >= 3 and left_key.startswith(right_key))
    )


def _outcome(markets: Iterable[dict], market_key: str, name: object) -> dict:
    for market in markets:
        if market.get("key") != market_key:
            continue
        for outcome in market.get("outcomes") or []:
            if outcome.get("name") == name:
                return outcome
        # Some feeds normalize team names differently between the event and
        # market objects.  Use the same conservative alias rule as scheduling.
        for outcome in market.get("outcomes") or []:
            if _same_team(outcome.get("name"), name):
                return outcome
    return {}


def _market(markets: Iterable[dict], market_key: str) -> dict:
    return next((market for market in markets if market.get("key") == market_key), {})


def _event_observed_at(event: dict, bookmakers: list[dict]) -> str | None:
    timestamps = []
    for bookmaker in bookmakers:
        value = bookmaker.get("last_update") or bookmaker.get("updated_at")
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(parsed):
            timestamps.append(parsed)
    if timestamps:
        return max(timestamps).isoformat()
    value = event.get("last_update") or event.get("updated_at")
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return parsed.isoformat() if pd.notna(parsed) else None


def to_cfbd_line_payload(
    events: Iterable[dict],
    games: pd.DataFrame,
    *,
    season: int,
    observed_at: str | None = None,
) -> list[dict]:
    """Map ParlayAPI events to the project's canonical intermediate payload.

    Only events that match exactly one scheduled game are returned.  Provider
    event IDs and freshness metadata are retained for the normalized snapshot
    layer; no synthetic game IDs are created.
    """
    schedule = games[games["season"].eq(season)].copy() if not games.empty else games
    if schedule.empty:
        return []
    captured = observed_at or datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for event in events:
        home = event.get("home_team") or event.get("home")
        away = event.get("away_team") or event.get("away")
        game = match_scheduled_game(schedule, home, away, event.get("commence_time"))
        if game is None:
            logger.warning("Skipping unmatched/ambiguous ParlayAPI event: %s at %s", away, home)
            continue

        bookmakers = event.get("bookmakers") or event.get("sources") or []
        if isinstance(bookmakers, dict):
            bookmakers = [dict(value, key=key) if isinstance(value, dict) else {"key": key} for key, value in bookmakers.items()]
        event_observed_at = _event_observed_at(event, bookmakers) or captured
        lines: list[dict] = []
        for bookmaker in bookmakers:
            markets = bookmaker.get("markets") or []
            home_ml = _outcome(markets, "h2h", home)
            away_ml = _outcome(markets, "h2h", away)
            home_spread = _outcome(markets, "spreads", home)
            away_spread = _outcome(markets, "spreads", away)
            over = _outcome(markets, "totals", "Over")
            under = _outcome(markets, "totals", "Under")
            quote = {
                "provider": bookmaker.get("title") or bookmaker.get("name") or bookmaker.get("key") or "unknown",
                "provider_event_id": event.get("id") or event.get("event_id"),
                "provider_observed_at": bookmaker.get("last_update") or bookmaker.get("updated_at") or event_observed_at,
                "is_live": bool(event.get("is_live") or event.get("live")),
                "stale_seconds": bookmaker.get("stale_seconds", event.get("stale_seconds")),
                "topped_up": bookmaker.get("topped_up", event.get("topped_up")),
                "spread": home_spread.get("point"),
                "homeSpreadOdds": home_spread.get("price"),
                "awaySpreadOdds": away_spread.get("price"),
                "overUnder": over.get("point", under.get("point")),
                "overOdds": over.get("price"),
                "underOdds": under.get("price"),
                "homeMoneyline": home_ml.get("price"),
                "awayMoneyline": away_ml.get("price"),
            }
            if any(value is not None for key, value in quote.items() if key not in {
                "provider", "provider_event_id", "provider_observed_at", "is_live", "stale_seconds", "topped_up",
            }):
                lines.append(quote)
        if lines:
            rows.append({
                "id": game["game_id"],
                "season": season,
                "source": SOURCE,
                "provider_event_id": event.get("id") or event.get("event_id"),
                "provider_observed_at": event_observed_at,
                "available_at": captured,
                "is_live": bool(event.get("is_live") or event.get("live")),
                "lines": lines,
            })
    return rows
