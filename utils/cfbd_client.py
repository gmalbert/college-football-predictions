"""utils/cfbd_client.py — Thin wrapper around the College Football Data API (v5)."""
from __future__ import annotations
import cfbd
from functools import lru_cache
from utils.config import get_secret
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_config() -> cfbd.Configuration:
    """Build configuration using Bearer access_token (cfbd SDK v5+)."""
    config = cfbd.Configuration()
    config.access_token = get_secret("cfbd", "api_key")
    return config


@lru_cache(maxsize=1)
def _client() -> cfbd.ApiClient:
    return cfbd.ApiClient(_get_config())


# ── Games ─────────────────────────────────────────────────────────

def get_games(year: int, season_type: str = "regular", week: int | None = None) -> list:
    api = cfbd.GamesApi(_client())
    try:
        return api.get_games(
            year=year,
            season_type=cfbd.SeasonType(season_type),
            week=week,
        )
    except Exception as e:
        logger.error(f"CFBD get_games error: {e}")
        return []


def get_game_team_stats(
    year: int,
    week: int | None = None,
    season_type: str | None = None,
) -> list:
    api = cfbd.GamesApi(_client())
    try:
        kwargs: dict = {"year": year}
        if week is not None:
            kwargs["week"] = week
        if season_type is not None:
            kwargs["season_type"] = cfbd.SeasonType(season_type)
        return api.get_game_team_stats(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_game_team_stats error: {e}")
        return []


# ── Betting Lines ─────────────────────────────────────────────────

def get_lines(year: int, week: int | None = None) -> list:
    api = cfbd.BettingApi(_client())
    try:
        return api.get_lines(year=year, week=week)
    except Exception as e:
        logger.error(f"CFBD get_lines error: {e}")
        return []


# ── Advanced Stats ────────────────────────────────────────────────

def get_advanced_stats(year: int) -> list:
    api = cfbd.StatsApi(_client())
    try:
        return api.get_advanced_season_stats(year=year)
    except Exception as e:
        logger.error(f"CFBD get_advanced_stats error: {e}")
        return []


# ── Ratings ───────────────────────────────────────────────────────

def get_sp_ratings(year: int) -> list:
    api = cfbd.RatingsApi(_client())
    try:
        return api.get_sp(year=year)
    except Exception as e:
        logger.error(f"CFBD get_sp error: {e}")
        return []


def get_elo_ratings(year: int) -> list:
    api = cfbd.RatingsApi(_client())
    try:
        return api.get_elo(year=year)
    except Exception as e:
        logger.error(f"CFBD get_elo error: {e}")
        return []


# ── Recruiting ────────────────────────────────────────────────────

def get_team_recruiting(year: int) -> list:
    api = cfbd.RecruitingApi(_client())
    try:
        return api.get_team_recruiting_rankings(year=year)
    except Exception as e:
        logger.error(f"CFBD get_team_recruiting error: {e}")
        return []


# ── Rankings ──────────────────────────────────────────────────────

def get_rankings(year: int, week: int | None = None) -> list:
    api = cfbd.RankingsApi(_client())
    try:
        return api.get_rankings(year=year, week=week)
    except Exception as e:
        logger.error(f"CFBD get_rankings error: {e}")
        return []


# ── Teams ─────────────────────────────────────────────────────────

def get_teams(conference: str | None = None) -> list:
    api = cfbd.TeamsApi(_client())
    try:
        return api.get_teams(conference=conference)
    except Exception as e:
        logger.error(f"CFBD get_teams error: {e}")
        return []


# ── Talent ────────────────────────────────────────────────

def get_talent(year: int) -> list:
    api = cfbd.TeamsApi(_client())
    try:
        return api.get_talent(year=year)
    except Exception as e:
        logger.error(f"CFBD get_talent error: {e}")
        return []


def get_talent(year: int) -> list:
    api = cfbd.TeamsApi(_client())
    try:
        return api.get_talent(year=year)
    except Exception as e:
        logger.error(f"CFBD get_talent error: {e}")
        return []


# ── Advanced game-level stats ─────────────────────────────────────

def get_advanced_game_stats(year: int, season_type: str = "regular") -> list:
    api = cfbd.StatsApi(_client())
    try:
        return api.get_advanced_game_stats(
            year=year,
            season_type=cfbd.SeasonType(season_type),
        )
    except Exception as e:
        logger.error(f"CFBD get_advanced_game_stats error: {e}")
        return []
