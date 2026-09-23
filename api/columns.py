"""Column projections for the Parquet artifacts the API reads.

``feature_matrix.parquet`` carries 266 columns; the API references 67 of them.
That matters more than it sounds: pyarrow's decode costs roughly 2.6x the
decoded frame size, so reading all 266 columns cost 117 MB of resident memory
to produce a 45 MB frame — 47% of the API's entire data footprint, for columns
nothing reads. See ``scripts/probe_column_usage.py``.

Every list here is a superset: unknown columns are dropped at read time, so a
name only has to appear if something might use it.
"""
from __future__ import annotations

from utils.feature_engine import (
    SPREAD_FEATURES,
    TOTAL_COVER_FEATURES,
    TOTAL_FEATURES,
    WIN_FEATURES,
)
from utils.repo_audit import LEAKAGE_RISK_COLUMNS

__all__ = [
    "MODEL_FEATURES",
    "FEATURE_MATRIX_COLUMNS",
    "AUDIT_FEATURE_MATRIX_COLUMNS",
    "GAMES_COLUMNS",
    "LINE_SNAPSHOT_COLUMNS",
    "TEAM_GAME_STATS_COLUMNS",
]

# Everything predict_batch / predict_for_display needs.
MODEL_FEATURES: list[str] = list(
    dict.fromkeys(WIN_FEATURES + SPREAD_FEATURES + TOTAL_FEATURES + TOTAL_COVER_FEATURES)
)

# Identity, schedule, market and result columns the pages display or filter on.
_DISPLAY_COLUMNS: list[str] = [
    "game_id", "season", "week", "season_type", "start_date", "neutral_site",
    "home_team", "away_team", "home_conference", "away_conference",
    "conference_game", "home_score", "away_score", "home_margin",
    "total_points", "home_win", "prediction_scope",
    "market_spread", "market_total", "home_moneyline", "away_moneyline",
    "market_spread_open", "market_spread_move",
    "market_total_open", "market_total_move",
    "market_spread_dispersion", "market_spread_book_count",
    "market_total_dispersion", "market_total_book_count",
    "market_home_prob",
]

FEATURE_MATRIX_COLUMNS: list[str] = list(
    dict.fromkeys(MODEL_FEATURES + _DISPLAY_COLUMNS)
)

# The Data & Model Quality page audits the feature-matrix *artifact*, so it must
# be able to see the columns the audit reasons about. This is deliberately
# narrower than FEATURE_MATRIX_COLUMNS and derived from the audit's own
# constants rather than hand-listed, because getting it wrong is silent:
# utils/contracts.py skips checks whose columns are absent, so a missing column
# makes an audit that should warn report "pass" instead.
#
# That is not hypothetical — projecting this file with FEATURE_MATRIX_COLUMNS
# alone turned the unsafe_exploration_columns warning into a false all-clear,
# which the parity harness caught.
AUDIT_FEATURE_MATRIX_COLUMNS: list[str] = list(
    dict.fromkeys(
        [
            # FEATURE_CONTRACT.required / non_null / ranges
            "game_id",
            "season",
            "home_team",
            "away_team",
            # validate_feature_matrix's point-in-time check
            "feature_as_of",
            "start_date",
            # presence of these is what the leakage check reports on
            *LEAKAGE_RISK_COLUMNS,
        ]
    )
)

GAMES_COLUMNS: list[str] = [
    "game_id", "season", "week", "season_type", "start_date", "neutral_site",
    "home_team", "away_team", "home_conference", "away_conference",
    "conference_game", "home_score", "away_score", "home_margin",
    "total_points", "home_win", "completed",
]

LINE_SNAPSHOT_COLUMNS: list[str] = [
    "game_id", "sportsbook", "market", "side", "line", "odds",
    "captured_at", "available_at", "source", "is_live", "stale_seconds",
    "provider_observed_at", "topped_up",
]

TEAM_GAME_STATS_COLUMNS: list[str] = [
    "game_id", "season", "week", "team", "home_away", "opponent",
    "points", "total_yards", "rushing_yards", "passing_yards",
    "turnovers", "third_down_eff", "possession_minutes",
]
