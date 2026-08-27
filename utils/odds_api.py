"""Small, quota-aware adapter for The Odds API's current NCAAF markets."""
from __future__ import annotations

from typing import Iterable
import unicodedata

import pandas as pd
import requests

from utils.config import get_secret
from utils.logger import get_logger

logger = get_logger(__name__)

ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
TIMEOUT = 20


def _api_key() -> str:
    """Return the configured Odds API key without ever logging it."""
    key = get_secret("odds", "api_key").strip()
    if not key:
        raise ValueError("ODDS_API_KEY is empty.")
    return key


def is_configured() -> bool:
    """Whether an Odds API key is available, without exposing its value."""
    try:
        return bool(_api_key())
    except ValueError:
        return False


def get_ncaaf_odds() -> list[dict]:
    """Fetch all current US NCAAF main markets in one three-credit request."""
    try:
        response = requests.get(
            ODDS_URL,
            params={
                "apiKey": _api_key(),
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        logger.info(
            "Odds API NCAAF snapshot received; credits used=%s remaining=%s last=%s",
            response.headers.get("x-requests-used", "unknown"),
            response.headers.get("x-requests-remaining", "unknown"),
            response.headers.get("x-requests-last", "unknown"),
        )
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.error("Odds API NCAAF snapshot error: %s", exc)
        return []


def _team_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return "".join(character for character in normalized.casefold() if character.isalnum())


def _same_team(left: object, right: object) -> bool:
    """Match CFBD's school names to feeds that sometimes append a mascot."""
    left_key, right_key = _team_key(left), _team_key(right)
    return bool(left_key and right_key) and (
        left_key == right_key
        # FBS acronyms such as TCU, UAB, FIU and SMU need the same mascot
        # suffix matching as ordinary school names. Pairwise home/away matching
        # and the unique-candidate requirement keep this conservative.
        or (len(left_key) >= 3 and right_key.startswith(left_key))
        or (len(right_key) >= 3 and left_key.startswith(right_key))
    )


def match_scheduled_game(
    schedule: pd.DataFrame, home: object, away: object, start_date: object | None = None,
) -> pd.Series | None:
    """Return one CFBD game, using kickoff time to break same-name collisions."""
    candidates = schedule[
        schedule["home_team"].map(lambda value: _same_team(value, home))
        & schedule["away_team"].map(lambda value: _same_team(value, away))
    ]
    if len(candidates) == 1:
        return candidates.iloc[0]
    if len(candidates) > 1 and start_date is not None and "start_date" in candidates:
        target = pd.to_datetime(start_date, utc=True, errors="coerce")
        starts = pd.to_datetime(candidates["start_date"], utc=True, errors="coerce")
        if pd.notna(target) and starts.notna().any():
            nearest = (starts - target).abs().idxmin()
            if abs(starts.loc[nearest] - target) <= pd.Timedelta(hours=24):
                return candidates.loc[nearest]
    return None


def _outcome(markets: Iterable[dict], market_key: str, name: str) -> dict:
    for market in markets:
        if market.get("key") == market_key:
            for outcome in market.get("outcomes") or []:
                if outcome.get("name") == name:
                    return outcome
    return {}


def to_cfbd_line_payload(events: Iterable[dict], games: pd.DataFrame, *, season: int) -> list[dict]:
    """Map Odds API events to the CFBD-shaped payload consumed by this project.

    The market tables intentionally retain CFBD ``game_id`` values, so that a
    current quote joins the existing schedule and feature data without changing
    any downstream grains. Events that cannot be matched safely are excluded.
    """
    schedule = games.copy()
    if schedule.empty:
        return []
    schedule = schedule[schedule["season"].eq(season)].copy()
    rows: list[dict] = []
    for event in events:
        game = match_scheduled_game(
            schedule, event.get("home_team"), event.get("away_team"), event.get("commence_time")
        )
        if game is None:
            logger.warning(
                "Skipping unmatched/ambiguous Odds API event: %s at %s",
                event.get("away_team"), event.get("home_team"),
            )
            continue
        game_id = game["game_id"]
        lines: list[dict] = []
        for bookmaker in event.get("bookmakers") or []:
            markets = bookmaker.get("markets") or []
            home_spread = _outcome(markets, "spreads", event.get("home_team"))
            away_spread = _outcome(markets, "spreads", event.get("away_team"))
            over = _outcome(markets, "totals", "Over")
            under = _outcome(markets, "totals", "Under")
            home_moneyline = _outcome(markets, "h2h", event.get("home_team"))
            away_moneyline = _outcome(markets, "h2h", event.get("away_team"))
            quote = {
                "provider": bookmaker.get("title") or bookmaker.get("key"),
                "spread": home_spread.get("point"),
                "homeSpreadOdds": home_spread.get("price"),
                "awaySpreadOdds": away_spread.get("price"),
                "overUnder": over.get("point") or under.get("point"),
                "overOdds": over.get("price"),
                "underOdds": under.get("price"),
                "homeMoneyline": home_moneyline.get("price"),
                "awayMoneyline": away_moneyline.get("price"),
            }
            if any(value is not None for key, value in quote.items() if key != "provider"):
                lines.append(quote)
        if lines:
            rows.append({"id": game_id, "season": season, "lines": lines})
    return rows
