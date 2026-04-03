"""utils/cfbd_client.py — Thin wrapper around the College Football Data API."""
from __future__ import annotations
import cfbd
from cfbd.rest import ApiException
from functools import lru_cache
from utils.config import get_secret
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_config() -> cfbd.Configuration:
    config = cfbd.Configuration()
    config.api_key["Authorization"] = get_secret("cfbd", "api_key")
    config.api_key_prefix["Authorization"] = "Bearer"
    return config


@lru_cache(maxsize=1)
def _client() -> cfbd.ApiClient:
    return cfbd.ApiClient(_get_config())


# ── Games ─────────────────────────────────────────────────────────

def get_games(year: int, season_type: str = "regular", week: int | None = None) -> list:
    api = cfbd.GamesApi(_client())
    try:
        return api.get_games(year=year, season_type=season_type, week=week)
    except ApiException as e:
        logger.error(f"CFBD get_games error: {e}")
        return []


def get_game_team_stats(year: int, week: int | None = None) -> list:
    api = cfbd.GamesApi(_client())
    try:
        return api.get_team_game_stats(year=year, week=week)
    except ApiException as e:
        logger.error(f"CFBD get_team_game_stats error: {e}")
        return []


# ── Betting Lines ─────────────────────────────────────────────────

def get_lines(year: int, week: int | None = None) -> list:
    api = cfbd.BettingApi(_client())
    try:
        return api.get_lines(year=year, week=week)
    except ApiException as e:
        logger.error(f"CFBD get_lines error: {e}")
        return []


# ── Advanced Stats ────────────────────────────────────────────────

def get_advanced_stats(year: int) -> list:
    api = cfbd.StatsApi(_client())
    try:
        return api.get_advanced_team_season_stats(year=year)
    except ApiException as e:
        logger.error(f"CFBD get_advanced_stats error: {e}")
        return []


# ── Ratings ───────────────────────────────────────────────────────

def get_sp_ratings(year: int) -> list:
    api = cfbd.RatingsApi(_client())
    try:
        return api.get_sp_ratings(year=year)
    except ApiException as e:
        logger.error(f"CFBD get_sp_ratings error: {e}")
        return []


def get_elo_ratings(year: int) -> list:
    api = cfbd.RatingsApi(_client())
    try:
        return api.get_elo_ratings(year=year)
    except ApiException as e:
        logger.error(f"CFBD get_elo error: {e}")
        return []


# ── Recruiting ────────────────────────────────────────────────────

def get_team_recruiting(year: int) -> list:
    api = cfbd.RecruitingApi(_client())
    try:
        return api.get_recruiting_teams(year=year)
    except ApiException as e:
        logger.error(f"CFBD get_recruiting error: {e}")
        return []


# ── Rankings ──────────────────────────────────────────────────────

def get_rankings(year: int, week: int | None = None) -> list:
    api = cfbd.RankingsApi(_client())
    try:
        return api.get_rankings(year=year, week=week)
    except ApiException as e:
        logger.error(f"CFBD get_rankings error: {e}")
        return []


# ── Teams ─────────────────────────────────────────────────────────

def get_teams(conference: str | None = None) -> list:
    api = cfbd.TeamsApi(_client())
    try:
        return api.get_teams(conference=conference)
    except ApiException as e:
        logger.error(f"CFBD get_teams error: {e}")
        return []
