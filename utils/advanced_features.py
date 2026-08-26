"""Leakage-aware contextual feature transforms for college football.

Every transform is deterministic and accepts partially populated frames.  A
missing upstream source produces ``NaN`` plus coverage indicators; it is never
silently converted into an average team.  The registry at the bottom documents
the intended prediction-time contract for each feature family.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    family: str
    required_columns: tuple[str, ...]
    timing: str
    description: str


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna(False).astype(bool)


def _difference(frame: pd.DataFrame, home: str, away: str) -> pd.Series:
    return _numeric(frame, home) - _numeric(frame, away)


def build_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 50+ context, roster, matchup, and market-quality features."""
    result = frame.copy()
    week = _numeric(result, "week")
    season_type = result.get("season_type", pd.Series("regular", index=result.index))
    neutral = _boolean(result, "neutral_site")
    conference_game = _boolean(result, "conference_game")

    # Calendar and game context.
    result["season_progress"] = ((week - 1) / 15).clip(0, 1)
    result["early_season"] = (week <= 3).astype(float)
    result["late_season"] = (week >= 11).astype(float)
    result["postseason_flag"] = season_type.astype(str).str.lower().ne("regular").astype(float)
    result["neutral_site_flag"] = neutral.astype(float)
    result["conference_game_flag"] = conference_game.astype(float)
    result["cross_conference_flag"] = (
        result.get("home_conference", pd.Series(np.nan, index=result.index)).notna()
        & result.get("away_conference", pd.Series(np.nan, index=result.index)).notna()
        & result.get("home_conference", pd.Series(np.nan, index=result.index)).ne(
            result.get("away_conference", pd.Series(np.nan, index=result.index))
        )
    ).astype(float)

    # Poll expectations and trap spots. Lower ranks are stronger.
    home_rank = _numeric(result, "home_rank")
    away_rank = _numeric(result, "away_rank")
    next_opp_rank = _numeric(result, "home_next_opponent_rank")
    result["poll_rank_diff"] = away_rank - home_rank
    result["ranked_matchup"] = (home_rank.notna() & away_rank.notna()).astype(float)
    result["ranked_vs_unranked"] = (home_rank.notna() ^ away_rank.notna()).astype(float)
    result["home_trap_spot"] = (
        home_rank.notna() & away_rank.isna() & next_opp_rank.notna()
    ).astype(float)

    # Rest and schedule compression.
    home_rest = _numeric(result, "rest_days_home")
    away_rest = _numeric(result, "rest_days_away")
    result["rest_advantage"] = home_rest - away_rest
    result["home_short_week"] = (home_rest < 6).where(home_rest.notna()).astype(float)
    result["away_short_week"] = (away_rest < 6).where(away_rest.notna()).astype(float)
    result["home_bye_week"] = (home_rest >= 10).where(home_rest.notna()).astype(float)
    result["away_bye_week"] = (away_rest >= 10).where(away_rest.notna()).astype(float)

    # Travel, body-clock, and venue context.
    result["travel_miles_diff"] = _difference(result, "away_travel_miles", "home_travel_miles")
    result["away_time_zone_shift"] = _numeric(result, "away_time_zone_shift").abs()
    local_hour = _numeric(result, "kickoff_local_hour")
    result["away_body_clock_disadvantage"] = (
        result["away_time_zone_shift"] * ((local_hour < 13) | (local_hour >= 21)).astype(float)
    )
    elevation = _numeric(result, "elevation")
    away_home_elevation = _numeric(result, "away_home_elevation")
    result["altitude_acclimation_edge"] = (
        (elevation - away_home_elevation).clip(lower=0) / 1000.0
    )
    capacity = _numeric(result, "capacity").clip(lower=1)
    result["log_stadium_capacity"] = np.log(capacity)
    result["crowd_pressure"] = result["log_stadium_capacity"] * (~neutral).astype(float)
    result["dome_flag"] = _boolean(result, "is_dome").astype(float)
    result["grass_flag"] = _boolean(result, "is_grass").astype(float)

    # Weather and style interactions. Historical actual weather must only be
    # used when the prediction contract is kickoff-time; otherwise ingest a
    # timestamped forecast into the same columns.
    temperature = _numeric(result, "temperature")
    wind = _numeric(result, "wind_speed")
    precipitation = _numeric(result, "precipitation")
    result["freezing_weather"] = (temperature <= 32).where(temperature.notna()).astype(float)
    result["extreme_heat"] = (temperature >= 90).where(temperature.notna()).astype(float)
    result["high_wind"] = (wind >= 15).where(wind.notna()).astype(float)
    result["precipitation_flag"] = (precipitation > 0).where(precipitation.notna()).astype(float)
    result["adverse_weather"] = (
        (result["freezing_weather"].eq(1))
        | (result["high_wind"].eq(1))
        | (result["precipitation_flag"].eq(1))
    ).astype(float)
    avg_pass_rate = (
        _numeric(result, "home_off_pass_rate") + _numeric(result, "away_off_pass_rate")
    ) / 2
    result["wind_pass_interaction"] = wind * avg_pass_rate

    # Player, roster, and staff uncertainty.
    result["qb_availability_diff"] = _difference(
        result, "home_qb_availability", "away_qb_availability"
    )
    result["qb_uncertainty"] = (
        _numeric(result, "home_qb_uncertainty") + _numeric(result, "away_qb_uncertainty")
    ) / 2
    result["roster_continuity_diff"] = _difference(
        result, "home_roster_continuity", "away_roster_continuity"
    )
    result["coordinator_continuity_diff"] = _difference(
        result, "home_coordinator_continuity", "away_coordinator_continuity"
    )
    result["returning_production_diff"] = _difference(
        result, "home_ret_ppa_pct", "away_ret_ppa_pct"
    )
    result["portal_net_diff"] = _difference(result, "home_portal_net", "away_portal_net")
    result["recruiting_points_diff"] = _difference(
        result, "home_recruiting", "away_recruiting"
    )
    result["coach_tenure_diff"] = _difference(
        result, "home_coach_tenure", "away_coach_tenure"
    )

    # Regression-to-mean and matchup quality.
    result["turnover_regression_diff"] = -_difference(
        result, "home_turnover_margin_l5", "away_turnover_margin_l5"
    )
    result["explosive_balance_diff"] = _difference(
        result, "home_explosive_rate_l5", "away_explosive_rate_l5"
    )
    result["pace_diff"] = _difference(result, "home_plays_per_game_l5", "away_plays_per_game_l5")
    result["expected_possessions"] = (
        _numeric(result, "home_possessions_l5") + _numeric(result, "away_possessions_l5")
    ) / 2
    result["third_down_diff"] = _difference(
        result, "home_third_down_eff_l5", "away_third_down_eff_l5"
    )
    result["red_zone_diff"] = _difference(
        result, "home_red_zone_eff_l5", "away_red_zone_eff_l5"
    )
    result["pressure_rate_diff"] = _difference(
        result, "home_pressure_rate_l5", "away_pressure_rate_l5"
    )
    result["penalty_discipline_diff"] = -_difference(
        result, "home_penalty_yards_l5", "away_penalty_yards_l5"
    )
    result["special_teams_diff"] = _difference(
        result, "home_special_teams_epa_l5", "away_special_teams_epa_l5"
    )
    result["sack_rate_diff"] = _difference(result, "home_sack_rate_l5", "away_sack_rate_l5")
    result["scoring_drive_diff"] = _difference(
        result, "home_scoring_drive_pct", "away_scoring_drive_pct"
    )
    result["three_and_out_advantage"] = -_difference(
        result, "home_three_and_out_pct", "away_three_and_out_pct"
    )

    # Special game types and coverage confidence.
    result["rivalry_flag"] = _boolean(result, "is_rivalry").astype(float)
    result["bowl_flag"] = _boolean(result, "is_bowl").astype(float)
    result["rematch_flag"] = _boolean(result, "is_rematch").astype(float)
    home_class = result.get("home_division", pd.Series("", index=result.index)).astype(str)
    away_class = result.get("away_division", pd.Series("", index=result.index)).astype(str)
    result["fcs_mismatch"] = home_class.str.contains("FBS", case=False) ^ away_class.str.contains(
        "FBS", case=False
    )
    result["fcs_mismatch"] = result["fcs_mismatch"].astype(float)
    result["source_coverage_min"] = pd.concat(
        [
            _numeric(result, "home_source_coverage"),
            _numeric(result, "away_source_coverage"),
        ],
        axis=1,
    ).min(axis=1)

    # Market information quality. These are permitted only for a declared line
    # snapshot (open, current, or close), never an unlabelled mixed timestamp.
    result["market_line_move"] = _numeric(result, "current_line") - _numeric(result, "open_line")
    result["market_disagreement"] = _numeric(result, "book_dispersion")
    result["market_depth"] = _numeric(result, "sportsbook_count")
    result["stale_quote_hours"] = _numeric(result, "stale_quote_hours")

    monitored_inputs = [
        "home_qb_availability", "away_qb_availability", "home_roster_continuity",
        "away_roster_continuity", "temperature", "wind_speed", "away_travel_miles",
        "home_rank", "away_rank", "sportsbook_count",
    ]
    available = [column for column in monitored_inputs if column in result.columns]
    result["context_missing_count"] = (
        result[available].isna().sum(axis=1) if available else len(monitored_inputs)
    )
    result["context_coverage"] = 1.0 - result["context_missing_count"] / len(monitored_inputs)
    return result


FEATURE_REGISTRY: tuple[FeatureDefinition, ...] = (
    FeatureDefinition("season_progress", "calendar", ("week",), "schedule publication", "Normalized point in the season."),
    FeatureDefinition("early_season", "calendar", ("week",), "schedule publication", "Weeks 0-3 uncertainty regime."),
    FeatureDefinition("postseason_flag", "context", ("season_type",), "schedule publication", "Bowl/playoff regime."),
    FeatureDefinition("poll_rank_diff", "poll", ("home_rank", "away_rank"), "poll release", "Difference in current poll rank."),
    FeatureDefinition("home_trap_spot", "schedule", ("home_rank", "home_next_opponent_rank"), "poll release", "Ranked favorite before a ranked opponent."),
    FeatureDefinition("rest_advantage", "schedule", ("rest_days_home", "rest_days_away"), "prior final whistle", "Difference in days since previous game."),
    FeatureDefinition("away_body_clock_disadvantage", "travel", ("away_time_zone_shift", "kickoff_local_hour"), "schedule publication", "Time-zone and kickoff interaction."),
    FeatureDefinition("altitude_acclimation_edge", "venue", ("elevation", "away_home_elevation"), "venue publication", "Change in elevation for visiting team."),
    FeatureDefinition("crowd_pressure", "venue", ("capacity", "neutral_site"), "venue publication", "Log capacity suppressed at neutral sites."),
    FeatureDefinition("adverse_weather", "weather", ("temperature", "wind_speed", "precipitation"), "forecast captured_at", "Composite adverse forecast flag."),
    FeatureDefinition("wind_pass_interaction", "weather", ("wind_speed", "home_off_pass_rate", "away_off_pass_rate"), "forecast captured_at", "Wind exposure for pass-heavy teams."),
    FeatureDefinition("qb_availability_diff", "roster", ("home_qb_availability", "away_qb_availability"), "status available_at", "Starter/backup availability difference."),
    FeatureDefinition("qb_uncertainty", "roster", ("home_qb_uncertainty", "away_qb_uncertainty"), "status available_at", "Average quarterback status uncertainty."),
    FeatureDefinition("roster_continuity_diff", "roster", ("home_roster_continuity", "away_roster_continuity"), "preseason snapshot", "Returning snap-weighted continuity."),
    FeatureDefinition("coordinator_continuity_diff", "staff", ("home_coordinator_continuity", "away_coordinator_continuity"), "staff announcement", "Coordinator retention difference."),
    FeatureDefinition("turnover_regression_diff", "efficiency", ("home_turnover_margin_l5", "away_turnover_margin_l5"), "prior final whistle", "Contrarian turnover regression signal."),
    FeatureDefinition("expected_possessions", "pace", ("home_possessions_l5", "away_possessions_l5"), "prior final whistle", "Expected opportunity volume."),
    FeatureDefinition("market_line_move", "market", ("open_line", "current_line"), "quote captured_at", "Movement from open to declared snapshot."),
    FeatureDefinition("market_disagreement", "market", ("book_dispersion",), "quote captured_at", "Cross-book line range."),
    FeatureDefinition("context_coverage", "quality", (), "prediction time", "Share of optional context inputs observed."),
)
