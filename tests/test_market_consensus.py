"""Semantics of the snapshot-derived market consensus.

``build_market_consensus_from_snapshots`` feeds both the Streamlit pages and the
API, so the parity harness cannot catch a change in it — both sides would move
together. These cases pin the behaviour down directly, with values chosen so the
expected results are exact rather than approximate.

The function is also on the Weekly Predictions request path: it used to rebuild
by masking a frame per game and per book, which cost ~10s once the snapshot
history passed 200k quotes. See the comment in the implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.market import remove_vig  # noqa: E402
from utils.odds_ingestion import build_market_consensus_from_snapshots  # noqa: E402

T0 = "2026-09-01T12:00:00Z"
T1 = "2026-09-08T12:00:00Z"


def quote(game_id, book, market, side, line, captured_at, odds=-110, source="cfbd"):
    return {
        "game_id": game_id,
        "sportsbook": book,
        "market": market,
        "side": side,
        "line": line,
        "odds": odds,
        "captured_at": captured_at,
        "source": source,
    }


def schedule(*game_ids):
    return pd.DataFrame({"game_id": list(game_ids), "season": 2026})


def build(quotes, *game_ids, exclude_sources=()):
    frame = pd.DataFrame(quotes)
    return build_market_consensus_from_snapshots(
        frame, schedule(*game_ids), season=2026, exclude_sources=exclude_sources
    )


def row_for(consensus, game_id):
    match = consensus.loc[consensus["game_id"].eq(game_id)]
    assert len(match) == 1, f"expected one row for {game_id}, got {len(match)}"
    return match.iloc[0]


def test_current_is_the_latest_quote_and_opening_is_the_earliest():
    """Lines 3/4/5 closed against 2/3/4 opened, so the median moves 3.0 -> 4.0."""
    consensus = build(
        [
            quote(1, "book1", "spread", "home", 2.0, T0),
            quote(1, "book2", "spread", "home", 3.0, T0),
            quote(1, "book3", "spread", "home", 4.0, T0),
            quote(1, "book1", "spread", "home", 3.0, T1),
            quote(1, "book2", "spread", "home", 4.0, T1),
            quote(1, "book3", "spread", "home", 5.0, T1),
        ],
        1,
    )
    row = row_for(consensus, 1)
    assert row["market_spread"] == 4.0
    assert row["market_spread_open"] == 3.0
    assert row["market_spread_move"] == 1.0
    assert row["market_spread_book_count"] == 3
    # std of [3, 4, 5] with ddof=1 is exactly 1.0.
    assert row["market_spread_dispersion"] == 1.0


def test_total_falls_back_to_the_under_side():
    """A book quoting only the under still prices the total."""
    consensus = build(
        [
            quote(1, "book1", "total", "under", 47.0, T0),
            quote(1, "book1", "total", "under", 48.0, T1),
        ],
        1,
    )
    row = row_for(consensus, 1)
    assert row["market_total"] == 48.0
    assert row["market_total_open"] == 47.0
    assert row["market_total_book_count"] == 1
    # One quote cannot have a sample standard deviation.
    assert row["market_total_dispersion"] == 0.0


def test_over_wins_over_under_when_both_are_quoted():
    consensus = build(
        [
            quote(1, "book1", "total", "over", 50.0, T1),
            quote(1, "book1", "total", "under", 44.0, T1),
        ],
        1,
    )
    row = row_for(consensus, 1)
    assert row["market_total"] == 50.0


def test_zero_and_missing_prices_are_excluded_from_the_vig_consensus():
    """A placeholder price of 0 must not enter the no-vig consensus.

    It is still a reported moneyline — matching the previous behaviour — but it
    cannot contribute a probability, so the book count reflects only real pairs.
    """
    consensus = build(
        [
            quote(1, "book1", "moneyline", "home", 0.0, T1, odds=100),
            quote(1, "book1", "moneyline", "away", 0.0, T1, odds=-120),
            quote(1, "book2", "moneyline", "home", 0.0, T1, odds=0),
            quote(1, "book2", "moneyline", "away", 0.0, T1, odds=-110),
        ],
        1,
    )
    row = row_for(consensus, 1)
    assert row["home_moneyline"] == 50.0  # median of [100, 0]
    assert row["moneyline_book_count"] == 1  # only book1 has both prices
    assert row["market_home_prob"] == pytest.approx(remove_vig([100, -120])[0])


def test_a_book_quoting_only_a_spread_does_not_contribute_a_moneyline():
    consensus = build(
        [
            quote(1, "book1", "spread", "home", 3.0, T1),
            quote(1, "book2", "moneyline", "home", 0.0, T1, odds=-150),
            quote(1, "book2", "moneyline", "away", 0.0, T1, odds=130),
        ],
        1,
    )
    row = row_for(consensus, 1)
    assert row["home_moneyline"] == -150
    assert row["away_moneyline"] == 130
    assert row["moneyline_book_count"] == 1


def test_games_without_quotes_still_produce_a_row():
    consensus = build([quote(1, "book1", "spread", "home", 3.0, T1)], 1, 2)
    assert set(consensus["game_id"]) == {1}
    row = row_for(consensus, 1)
    assert row["market_spread"] == 3.0
    assert np.isnan(row["market_total"])
    assert np.isnan(row["home_moneyline"])
    assert row["market_total_book_count"] == 0


def test_season_filter_excludes_other_seasons():
    frame = pd.DataFrame([quote(1, "book1", "spread", "home", 3.0, T1)])
    other = pd.DataFrame({"game_id": [999], "season": 2025})
    consensus = build_market_consensus_from_snapshots(
        frame, other, season=2025, exclude_sources=()
    )
    assert consensus.empty


def test_excluded_sources_are_dropped():
    quotes = [
        quote(1, "book1", "spread", "home", 3.0, T1, source="cfbd"),
        quote(1, "book2", "spread", "home", 9.0, T1, source="parlay_api"),
    ]
    kept = build(quotes, 1, exclude_sources=("parlay_api",))
    assert row_for(kept, 1)["market_spread"] == 3.0

    all_in = build(quotes, 1, exclude_sources=())
    assert row_for(all_in, 1)["market_spread"] == 6.0  # median of [3, 9]


def test_missing_required_columns_returns_empty():
    frame = pd.DataFrame({"game_id": [1], "sportsbook": ["book1"]})
    consensus = build_market_consensus_from_snapshots(
        frame, schedule(1), season=2026, exclude_sources=()
    )
    assert consensus.empty
