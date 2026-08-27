"""Quota-aware pre-match NCAAF market adapter for TheRundown V2 API."""
from __future__ import annotations

from datetime import date, timedelta
import time

import pandas as pd
import requests

from utils.config import get_secret
from utils.logger import get_logger
from utils.odds_api import _same_team, match_scheduled_game

logger = get_logger(__name__)

BASE_URL = "https://therundown.io/api/v2/sports/1/events"
TIMEOUT = 20
# The free tier's three pre-match books: BetMGM, DraftKings and FanDuel.
AFFILIATE_NAMES = {"3": "BetMGM", "19": "DraftKings", "23": "FanDuel"}


def _api_key() -> str:
    key = get_secret("therundown", "api_key").strip()
    if not key:
        raise ValueError("THERUNDOWN_API_KEY is empty.")
    return key


def is_configured() -> bool:
    try:
        return bool(_api_key())
    except ValueError:
        return False


def get_ncaaf_events(*, start: date | None = None, days: int = 8) -> list[dict]:
    """Fetch the next eight NCAAF calendars at a 1 request/sec-safe cadence."""
    events: list[dict] = []
    first_day = start or date.today()
    try:
        key = _api_key()
        for offset in range(days):
            response = requests.get(
                f"{BASE_URL}/{first_day + timedelta(days=offset)}",
                headers={"X-TheRundown-Key": key},
                params={
                    "market_ids": "1,2,3",
                    "affiliate_ids": ",".join(AFFILIATE_NAMES),
                    "main_line": "true",
                    "hide_closed": "true",
                    "offset": "240",
                },
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            events.extend(response.json().get("events", []))
            logger.info(
                "TheRundown %s received; data points=%s remaining=%s",
                first_day + timedelta(days=offset),
                response.headers.get("X-Datapoints", "unknown"),
                response.headers.get("X-Datapoints-Remaining", "unknown"),
            )
            if offset < days - 1:
                time.sleep(1.05)
        return events
    except (requests.RequestException, ValueError) as exc:
        logger.error("TheRundown NCAAF snapshot error: %s", exc)
        return []


def to_cfbd_line_payload(events: list[dict], games: pd.DataFrame, *, season: int) -> list[dict]:
    """Map TheRundown's event/market hierarchy to this app's canonical grain."""
    schedule = games[games["season"].eq(season)].copy() if not games.empty else games
    rows: list[dict] = []
    for event in events:
        teams = event.get("teams") or []
        away = next((team.get("name") for team in teams if team.get("is_away")), None)
        home = next((team.get("name") for team in teams if team.get("is_home")), None)
        game = match_scheduled_game(schedule, home, away, event.get("event_date"))
        if game is None:
            logger.warning("Skipping unmatched/ambiguous TheRundown event: %s at %s", away, home)
            continue
        quotes: dict[str, dict] = {}
        for market in event.get("markets") or []:
            market_id = market.get("market_id")
            if market.get("period_id") != 0 or market_id not in {1, 2, 3}:
                continue
            for participant in market.get("participants") or []:
                name = participant.get("name")
                for line in participant.get("lines") or []:
                    for affiliate_id, price in (line.get("prices") or {}).items():
                        value = price.get("price")
                        if value in (None, 0.0001):
                            continue
                        quote = quotes.setdefault(
                            str(affiliate_id), {"provider": AFFILIATE_NAMES.get(str(affiliate_id), str(affiliate_id))}
                        )
                        if market_id == 1:
                            if _same_team(name, home): quote["homeMoneyline"] = value
                            elif _same_team(name, away): quote["awayMoneyline"] = value
                        elif market_id == 2:
                            if _same_team(name, home):
                                quote["spread"] = line.get("value")
                                quote["homeSpreadOdds"] = value
                            elif _same_team(name, away): quote["awaySpreadOdds"] = value
                        elif market_id == 3:
                            quote["overUnder"] = line.get("value")
                            if str(name).casefold() == "over": quote["overOdds"] = value
                            elif str(name).casefold() == "under": quote["underOdds"] = value
        lines = [quote for quote in quotes.values() if len(quote) > 1]
        if lines:
            rows.append({"id": game["game_id"], "season": season, "lines": lines})
    return rows
