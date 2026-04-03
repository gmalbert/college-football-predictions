"""utils/espn_client.py — ESPN public API wrapper."""
from __future__ import annotations
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

ESPN_BASE = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
)
TIMEOUT = 15


def _get(endpoint: str, params: dict | None = None) -> dict | None:
    url = f"{ESPN_BASE}/{endpoint}"
    try:
        resp = requests.get(url, params=params or {}, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"ESPN API error ({endpoint}): {e}")
        return None


def get_scoreboard(limit: int = 50, groups: int = 80) -> list[dict]:
    """Fetch live / recent college football scores."""
    data = _get("scoreboard", {"limit": limit, "groups": groups})
    if not data:
        return []
    games = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        home = comp["competitors"][0]
        away = comp["competitors"][1]
        games.append({
            "game_id": event["id"],
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_score": home.get("score", "0"),
            "away_score": away.get("score", "0"),
            "status": event["status"]["type"]["shortDetail"],
            "home_logo": home["team"].get("logo"),
            "away_logo": away["team"].get("logo"),
        })
    return games


def get_team_roster(team_id: int) -> list[dict]:
    """Return the current roster for a given ESPN team ID."""
    data = _get(f"teams/{team_id}/roster")
    if not data:
        return []
    roster = []
    for group in data.get("athletes", []):
        for player in group.get("items", []):
            roster.append({
                "name": player["displayName"],
                "position": player.get("position", {}).get("abbreviation"),
                "jersey": player.get("jersey"),
                "year": player.get("experience", {}).get("displayValue"),
            })
    return roster


def get_rankings() -> list[dict]:
    """Fetch current poll rankings from ESPN."""
    data = _get("rankings")
    if not data:
        return []
    polls = []
    for poll in data.get("rankings", []):
        for rank in poll.get("ranks", []):
            polls.append({
                "poll": poll["name"],
                "rank": rank["current"],
                "team": rank["team"]["location"],
                "record": rank.get("recordSummary", ""),
            })
    return polls
