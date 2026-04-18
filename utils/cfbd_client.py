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


# ── Weather ───────────────────────────────────────────────────────

def get_game_weather(year: int, week: int | None = None, season_type: str = "regular") -> list:
    """Get weather data for games in a given year/week."""
    api = cfbd.GamesApi(_client())
    try:
        kwargs: dict = {"year": year, "season_type": cfbd.SeasonType(season_type)}
        if week is not None:
            kwargs["week"] = week
        return api.get_weather(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_game_weather error: {e}")
        return []


# ── Venues ────────────────────────────────────────────────────────

def get_venues() -> list:
    """Get all college football venue information."""
    api = cfbd.VenuesApi(_client())
    try:
        return api.get_venues()
    except Exception as e:
        logger.error(f"CFBD get_venues error: {e}")
        return []


# ── FPI Ratings ───────────────────────────────────────────────────

def get_fpi_ratings(year: int) -> list:
    """Get ESPN FPI ratings for a given year."""
    api = cfbd.RatingsApi(_client())
    try:
        return api.get_fpi(year=year)
    except Exception as e:
        logger.error(f"CFBD get_fpi_ratings error: {e}")
        return []


# ── SRS Ratings ───────────────────────────────────────────────────

def get_srs_ratings(year: int) -> list:
    """Get Simple Rating System (SRS) ratings for a given year."""
    api = cfbd.RatingsApi(_client())
    try:
        return api.get_srs(year=year)
    except Exception as e:
        logger.error(f"CFBD get_srs_ratings error: {e}")
        return []


# ── Pre-Game Win Probabilities ────────────────────────────────────

def get_pregame_win_prob(year: int, week: int | None = None) -> list:
    """Get CFBD's pre-game win probabilities."""
    api = cfbd.MetricsApi(_client())
    try:
        kwargs: dict = {"year": year}
        if week is not None:
            kwargs["week"] = week
        return api.get_pregame_win_probabilities(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_pregame_win_prob error: {e}")
        return []


# ── PPA / WEPA ────────────────────────────────────────────────────

def get_ppa_teams(year: int, team: str | None = None,
                  conference: str | None = None,
                  excl_garbage_time: bool = True) -> list:
    """Get team PPA (predicted points added) averages for a season."""
    api = cfbd.MetricsApi(_client())
    try:
        kwargs: dict = {"year": year, "exclude_garbage_time": excl_garbage_time}
        if team:
            kwargs["team"] = team
        if conference:
            kwargs["conference"] = conference
        return api.get_predicted_points_added_by_team(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_ppa_teams error: {e}")
        return []


def get_wepa_team_season(year: int, team: str | None = None) -> list:
    """Get opponent-adjusted (WEPA) team season stats."""
    api = cfbd.MetricsApi(_client())
    try:
        kwargs: dict = {"year": year}
        if team:
            kwargs["team"] = team
        # WEPA (adjusted team season stats) not available in this cfbd SDK version
        return []
    except Exception as e:
        logger.error(f"CFBD get_wepa_team_season error: {e}")
        return []


# ── Returning Production ──────────────────────────────────────────

def get_returning_production(year: int) -> list:
    """Get player returning production metrics for a given year."""
    api = cfbd.PlayersApi(_client())
    try:
        return api.get_returning_production(year=year)
    except Exception as e:
        logger.error(f"CFBD get_returning_production error: {e}")
        return []


# ── Transfer Portal ───────────────────────────────────────────────

def get_transfer_portal(year: int) -> list:
    """Get transfer portal data for a given year."""
    api = cfbd.PlayersApi(_client())
    try:
        return api.get_transfer_portal(year=year)
    except Exception as e:
        logger.error(f"CFBD get_transfer_portal error: {e}")
        return []


# ── Game Media ────────────────────────────────────────────────────

def get_game_media(year: int, week: int | None = None) -> list:
    """Get game media/broadcast information."""
    api = cfbd.GamesApi(_client())
    try:
        kwargs: dict = {"year": year}
        if week is not None:
            kwargs["week"] = week
        return api.get_media(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_game_media error: {e}")
        return []


# ── Play-by-Play ──────────────────────────────────────────────────

def get_plays(year: int, week: int, season_type: str = "regular",
              team: str | None = None) -> list:
    """Get play-by-play data for a specific year/week."""
    api = cfbd.PlaysApi(_client())
    try:
        kwargs: dict = {
            "year": year,
            "week": week,
            "season_type": cfbd.SeasonType(season_type),
        }
        if team:
            kwargs["team"] = team
        return api.get_plays(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_plays error: {e}")
        return []


# ── Drives ────────────────────────────────────────────────────────

def get_drives(year: int, week: int, season_type: str = "regular",
               team: str | None = None) -> list:
    """Get drive-by-drive data."""
    api = cfbd.DrivesApi(_client())
    try:
        kwargs: dict = {
            "year": year,
            "week": week,
            "season_type": cfbd.SeasonType(season_type),
        }
        if team:
            kwargs["team"] = team
        return api.get_drives(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_drives error: {e}")
        return []


# ── Player Usage ──────────────────────────────────────────────────

def get_player_usage(year: int, team: str | None = None) -> list:
    """Get player usage metrics (snap/play share, PPA contribution)."""
    api = cfbd.PlayersApi(_client())
    try:
        kwargs: dict = {"year": year}
        if team:
            kwargs["team"] = team
        return api.get_player_usage(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_player_usage error: {e}")
        return []


# ── Game Player Stats ─────────────────────────────────────────────

def get_game_player_stats(year: int, week: int | None = None,
                          season_type: str = "regular",
                          team: str | None = None) -> list:
    """Get player statistics by game."""
    api = cfbd.GamesApi(_client())
    try:
        kwargs: dict = {
            "year": year,
            "season_type": cfbd.SeasonType(season_type),
        }
        if week:
            kwargs["week"] = week
        if team:
            kwargs["team"] = team
        return api.get_game_player_stats(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_game_player_stats error: {e}")
        return []


# ── Coaches ───────────────────────────────────────────────────────

def get_coaches(year: int | None = None, team: str | None = None) -> list:
    """Get coaching history and records."""
    api = cfbd.CoachesApi(_client())
    try:
        kwargs: dict = {}
        if year:
            kwargs["year"] = year
        if team:
            kwargs["team"] = team
        return api.get_coaches(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_coaches error: {e}")
        return []


# ── Team Matchup ──────────────────────────────────────────────────

def get_team_matchup(team1: str, team2: str,
                     min_year: int | None = None,
                     max_year: int | None = None) -> dict:
    """Get head-to-head matchup history between two teams."""
    api = cfbd.TeamsApi(_client())
    try:
        kwargs: dict = {"team1": team1, "team2": team2}
        if min_year:
            kwargs["min_year"] = min_year
        if max_year:
            kwargs["max_year"] = max_year
        return api.get_matchup(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_team_matchup error: {e}")
        return {}


# ── In-Game Win Probability ───────────────────────────────────────

def get_win_probability_chart(game_id: int) -> list:
    """Get play-by-play win probability data for a completed game."""
    api = cfbd.MetricsApi(_client())
    try:
        return api.get_win_probability(game_id=game_id)
    except Exception as e:
        logger.error(f"CFBD get_win_probability_chart error: {e}")
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
