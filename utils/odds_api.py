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

# Market feeds use a mixture of abbreviations, historical school names, and
# mascotted names while CFBD generally uses one schedule-facing label. Keep
# these aliases conservative and explicit: they are applied before the
# existing exact/prefix matcher, so they cannot turn an arbitrary fuzzy match
# into a game assignment.
_TEAM_ALIAS_GROUPS = (
    ("appstate", "appalachianstate"),
    ("wpi", "worcesterpolytechnic"),
    ("newengland", "universityofnewengland"),
    ("vermontstatecastleton", "castletonstatecollege"),
    ("wheatonil", "wheatoncollegeillinois"),
    ("ualbany", "albany"),
    ("longislanduniversity", "longisland"),
    ("northcarolinawesleyan", "ncwesleyan"),
    ("uncpembroke", "northcarolinapembroke"),
    ("westminsterpa", "westminstercollegepa"),
    ("utmartin", "tennesseemartin"),
    ("maryvilletn", "maryvillecollege"),
    ("wilmingtonoh", "wilmingtoncollege"),
    ("lakeland", "lakelanduniversity"),
    ("callutheran", "californialutheran"),
    ("augustanasd", "augustanauniversitysd"),
    ("saintxavier", "stxavieril"),
    ("augustanail", "augustanacollegeil"),
    ("cornellia", "cornellcollegeia"),
    ("stthomasmn", "stthomasminnesota"),
    ("washingtonstlouis", "washingtonmissouri"),
    ("csupueblo", "coloradostatepueblo"),
    ("saintanselm", "stanselm"),
    ("washingtonandlee", "washingtonleeuniversity"),
    ("westminstermo", "westminstercollegemo"),
    ("eureka", "eurekacollege"),
    ("iupennsylvania", "indianapa"),
    ("simpsonia", "simpsoncollege"),
    ("northwesternmn", "univofnorthwesternstpaul"),
    ("utpermianbasin", "universityoftexasofthepermianbasin"),
    ("trinitytx", "trinityuniversitytx"),
    ("trinityct", "trinitycollegect"),
    ("loras", "lorascollege"),
    ("colby", "colbycollege"),
    ("benedictineil", "benedictineuniversityil"),
    ("easttexasam", "texasamcommerce"),
    ("concordiamn", "concordiamoorhead"),
    ("randolphmacon", "randolphmaconcollege"),
    ("wesleyanct", "wesleyanuniversityct"),
    ("saintjohnsmn", "stjohnsmn"),
    ("lyon", "lyoncollege"),
    ("utriograndevalley", "utrgv"),
    ("ucf", "centralflorida"),
    ("louisiana", "ullafayette"),
    ("selouisiana", "southeasternlouisiana"),
    ("ulmonroe", "louisianamonroe"),
    ("massachusetts", "umass", "umassminutemen"),
    ("olemiss", "mississippi"),
)


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
        # ``requests`` may include the full query URL, including the API key.
        logger.error("Odds API NCAAF snapshot error: %s", type(exc).__name__)
        return []


def _team_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    key = "".join(character for character in normalized.casefold() if character.isalnum())
    for canonical, *aliases in _TEAM_ALIAS_GROUPS:
        if key == canonical or key in aliases:
            return canonical
    return key


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


def match_market_event(
    schedule: pd.DataFrame, home: str, away: str, start_date: object | None = None,
) -> tuple[pd.Series, bool] | None:
    """Match an event and identify whether its provider uses the reverse orientation."""
    if schedule.empty or not home or not away:
        return None
    direct = match_scheduled_game(schedule, home, away, start_date)
    reverse = match_scheduled_game(schedule, away, home, start_date)
    if direct is None:
        return (reverse, True) if reverse is not None else None
    if reverse is None:
        return direct, False

    def rank(game: pd.Series, provider_home: str, provider_away: str) -> tuple[int, float]:
        kickoff = pd.to_datetime(game.get("start_date"), utc=True, errors="coerce")
        provider_kickoff = pd.to_datetime(start_date, utc=True, errors="coerce")
        hours = abs((kickoff - provider_kickoff).total_seconds()) if pd.notna(kickoff) and pd.notna(provider_kickoff) else float("inf")
        exact_matches = (
            int(_team_key(game["home_team"]) == _team_key(provider_home))
            + int(_team_key(game["away_team"]) == _team_key(provider_away))
        )
        return exact_matches, -hours

    direct_rank, reverse_rank = rank(direct, home, away), rank(reverse, away, home)
    if direct_rank == reverse_rank:
        return None
    return (direct, False) if direct_rank > reverse_rank else (reverse, True)


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
        home, away = event.get("home_team"), event.get("away_team")
        match = match_market_event(schedule, home, away, event.get("commence_time"))
        if match is None:
            logger.warning(
                "Skipping unmatched/ambiguous Odds API event: %s at %s",
                event.get("away_team"), event.get("home_team"),
            )
            continue
        game, reversed_teams = match
        if reversed_teams:
            home, away = away, home
        game_id = game["game_id"]
        lines: list[dict] = []
        for bookmaker in event.get("bookmakers") or []:
            markets = bookmaker.get("markets") or []
            home_spread = _outcome(markets, "spreads", home)
            away_spread = _outcome(markets, "spreads", away)
            over = _outcome(markets, "totals", "Over")
            under = _outcome(markets, "totals", "Under")
            home_moneyline = _outcome(markets, "h2h", home)
            away_moneyline = _outcome(markets, "h2h", away)
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
