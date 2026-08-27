from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.advanced_features import build_context_features
from utils.betting import generate_moneyline_pick, generate_spread_pick
from utils.contracts import safe_merge, validate_feature_matrix, validate_games
from utils.feature_observations import attach_game_observations
from utils.prospective_ledger import select_quote_as_of
from utils.evaluation import (
    calibration_table,
    conformal_interval,
    evaluate_release_gates,
    interval_metrics,
    probability_metrics,
)
from utils.joint_scores import JointScoreDistribution, fit_residual_distribution
from utils.market import (
    BetResult,
    Market,
    Side,
    american_to_decimal,
    closing_line_value,
    consensus_quotes,
    line_movement_features,
    remove_vig,
    settle_bet,
    spread_edge,
)
from utils.odds_ingestion import build_market_consensus, normalize_cfbd_line_snapshots
from utils.risk import RiskLimits, conservative_probability, size_portfolio
from utils.seasons import current_cfb_season, rolling_season_window
from utils.temporal import (
    add_rest_features,
    point_in_time_join,
    rolling_team_features,
    to_team_game_long,
    walk_forward_season_splits,
)
from utils.challenger_models import MarketBaselineClassifier, MarketBaselineRegressor
import utils.fetch_historical as fetch_historical
import utils.cfbd_client as cfbd_client


class _ConstantClassifier:
    def predict_proba(self, frame):
        return np.tile([0.4, 0.6], (len(frame), 1))


class _ConstantRegressor:
    def predict(self, frame):
        return np.full(len(frame), 7.0)


class IngestionTests(unittest.TestCase):
    def test_cfbd_key_normalizes_bearer_prefix_and_rejects_empty(self):
        with patch.object(cfbd_client, "get_secret", return_value="Bearer token-value"):
            self.assertEqual(cfbd_client._api_key(), "token-value")
        with patch.object(cfbd_client, "get_secret", return_value="   "):
            with self.assertRaisesRegex(ValueError, "CFBD_API_KEY is empty"):
                cfbd_client._api_key()

    def test_partial_partition_upsert_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            processed = Path(directory)
            existing = pd.DataFrame(
                {"game_id": [1, 2], "season": [2025, 2026], "value": [10, 20]}
            )
            existing.to_parquet(processed / "sample.parquet", index=False)

            def save_local(frame, name):
                frame.to_parquet(processed / f"{name}.parquet", index=False)

            refreshed = pd.DataFrame(
                {"game_id": [2, 3], "season": [2026, 2026], "value": [99, 30]}
            )
            with patch.object(fetch_historical, "PROCESSED_DIR", processed), patch.object(
                fetch_historical, "save_parquet", side_effect=save_local
            ):
                result = fetch_historical._save_processed_upsert(
                    refreshed, "sample", keys=["game_id"]
                )
            self.assertEqual(set(result.game_id), {1, 2, 3})
            self.assertEqual(result.set_index("game_id").loc[1, "value"], 10)
            self.assertEqual(result.set_index("game_id").loc[2, "value"], 99)

    def test_weather_batches_games_by_venue_range(self):
        with tempfile.TemporaryDirectory() as directory:
            processed = Path(directory)
            pd.DataFrame(
                {
                    "game_id": [1, 2],
                    "season": [2024, 2025],
                    "week": [1, 1],
                    "venue": ["Stadium", "Stadium"],
                    "start_date": ["2024-09-01T18:00:00Z", "2025-09-01T18:00:00Z"],
                }
            ).to_parquet(processed / "games.parquet", index=False)
            pd.DataFrame(
                {
                    "name": ["Stadium"],
                    "latitude": [40.0],
                    "longitude": [-75.0],
                    "dome": [False],
                }
            ).to_parquet(processed / "venues.parquet", index=False)

            response = {
                "hourly": {
                    "time": ["2024-09-01T18:00", "2025-09-01T18:00"],
                    "temperature_2m": [70, 72],
                    "wind_speed_10m": [5, 6],
                    "precipitation": [0, 0],
                    "weathercode": [0, 1],
                    "relativehumidity_2m": [50, 55],
                }
            }
            with patch.object(fetch_historical, "PROCESSED_DIR", processed), \
                    patch.object(fetch_historical, "save_parquet") as save, \
                    patch("requests.get", return_value=type(
                        "Response", (), {
                            "raise_for_status": lambda self: None,
                            "json": lambda self: response,
                        }
                    )()) as get:
                save.side_effect = lambda frame, name: frame.to_parquet(
                    processed / f"{name}.parquet", index=False
                )
                fetch_historical._build_weather(force=True)

            self.assertEqual(get.call_count, 1)
            params = get.call_args.args[0].split("?", 1)[1]
            self.assertIn("start_date=2024-09-01", params)
            self.assertIn("end_date=2025-09-01", params)


class MarketTests(unittest.TestCase):
    def test_market_anchored_wrappers_fallback_when_prices_missing(self):
        frame = pd.DataFrame(
            {"market_home_prob": [0.7, np.nan], "market_spread": [-3.5, np.nan]}
        )
        classifier = MarketBaselineClassifier(
            _ConstantClassifier(), tuple(frame.columns)
        )
        np.testing.assert_allclose(classifier.predict_proba(frame)[:, 1], [0.7, 0.6])
        regressor = MarketBaselineRegressor(
            _ConstantRegressor(), tuple(frame.columns), "market_spread", -1.0
        )
        np.testing.assert_allclose(regressor.predict(frame), [3.5, 7.0])
    def test_spread_sign_convention(self):
        edge = spread_edge(model_home_margin=10, home_spread=-7)
        self.assertEqual(edge.side, Side.HOME)
        self.assertEqual(edge.edge_points, 3)
        recommendation = generate_spread_pick("Home", "Away", 10, -7)
        self.assertEqual(recommendation.edge, 3)
        self.assertIn("Home", recommendation.pick)

    def test_spread_away_edge(self):
        edge = spread_edge(model_home_margin=3, home_spread=-7)
        self.assertEqual(edge.side, Side.AWAY)
        self.assertEqual(edge.edge_points, -4)

    def test_odds_and_devig(self):
        self.assertAlmostEqual(american_to_decimal(-110), 1.9090909, places=6)
        fair = remove_vig([-110, -110])
        np.testing.assert_allclose(fair, [0.5, 0.5])
        self.assertAlmostEqual(float(fair.sum()), 1.0)

    def test_moneyline_edge_uses_no_vig_market(self):
        recommendation = generate_moneyline_pick("Home", "Away", 0.62, -150, 130)
        self.assertIsNotNone(recommendation)
        self.assertIn("Home", recommendation.pick)

    def test_spread_and_total_settlement(self):
        home = settle_bet(
            market=Market.SPREAD, side=Side.HOME, home_score=28, away_score=20,
            line=-7, odds=-110, stake=11,
        )
        self.assertEqual(home.result, BetResult.WIN)
        self.assertAlmostEqual(home.profit, 10)
        push = settle_bet(
            market="spread", side="home", home_score=27, away_score=20,
            line=-7, stake=1,
        )
        self.assertEqual(push.result, BetResult.PUSH)
        under = settle_bet(
            market="total", side="under", home_score=21, away_score=17,
            line=42.5, stake=1,
        )
        self.assertEqual(under.result, BetResult.WIN)

    def test_quote_consensus_and_movement(self):
        quotes = pd.DataFrame(
            {
                "game_id": [1, 1, 1, 1],
                "sportsbook": ["A", "B", "A", "B"],
                "market": ["spread"] * 4,
                "side": ["home"] * 4,
                "captured_at": [
                    "2026-09-01T12:00Z", "2026-09-01T12:00Z",
                    "2026-09-02T12:00Z", "2026-09-02T12:00Z",
                ],
                "line": [-6.5, -7, -7.5, -8],
                "odds": [-110, -105, -108, -110],
            }
        )
        consensus = consensus_quotes(quotes.iloc[2:])
        self.assertEqual(consensus.iloc[0]["book_count"], 2)
        movement = line_movement_features(quotes)
        self.assertAlmostEqual(movement.iloc[0]["line_move"], -1.25)
        self.assertGreater(closing_line_value(market="spread", taken_line=-6.5, closing_line=-7), 0)

    def test_cfbd_quote_normalization(self):
        payload = [
            {
                "id": 7,
                "lines": [
                    {
                        "provider": "Book A", "spread": -7.5, "overUnder": 52.5,
                        "homeMoneyline": -280, "awayMoneyline": 230,
                    }
                ],
            }
        ]
        snapshots = normalize_cfbd_line_snapshots(
            payload, captured_at="2026-08-20T12:00Z"
        )
        self.assertEqual(len(snapshots), 6)
        home = snapshots.query("market == 'spread' and side == 'home'").iloc[0]
        away = snapshots.query("market == 'spread' and side == 'away'").iloc[0]
        self.assertEqual(home["line"], -7.5)
        self.assertEqual(away["line"], 7.5)

    def test_provider_consensus_preserves_open_move_and_devigs_per_book(self):
        payload = [
            {
                "id": 8,
                "lines": [
                    {
                        "provider": "Book A", "spread": -7, "spreadOpen": -6.5,
                        "overUnder": 52, "overUnderOpen": 51,
                        "homeMoneyline": -280, "awayMoneyline": 230,
                    },
                    {
                        "provider": "Book B", "spread": -7.5, "spreadOpen": -7,
                        "overUnder": 53, "overUnderOpen": 51.5,
                        "homeMoneyline": -300, "awayMoneyline": 240,
                    },
                ],
            }
        ]
        consensus = build_market_consensus(payload).iloc[0]
        self.assertEqual(consensus["market_spread"], -7.25)
        self.assertEqual(consensus["market_spread_open"], -6.75)
        self.assertEqual(consensus["market_spread_move"], -0.5)
        self.assertEqual(consensus["market_total_move"], 1.25)
        self.assertEqual(consensus["market_total_book_count"], 2)
        self.assertTrue(0 < consensus["market_home_prob"] < 1)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame(
            {
                "game_id": [1, 2, 3, 4],
                "season": [2024] * 4,
                "week": [1, 2, 3, 4],
                "start_date": pd.to_datetime(
                    ["2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22"], utc=True
                ),
                "home_team": ["A", "C", "A", "B"],
                "away_team": ["B", "A", "C", "A"],
                "home_score": [21, 10, 24, 14],
                "away_score": [14, 17, 20, 28],
                "neutral_site": [False] * 4,
            }
        )

    def test_rest_tracks_both_sides(self):
        enriched = add_rest_features(self.games)
        game_three = enriched.set_index("game_id").loc[3]
        self.assertEqual(game_three["rest_days_home"], 7)

    def test_rolling_excludes_current_game(self):
        long = to_team_game_long(self.games)
        rolled = rolling_team_features(long, value_columns=["points_for"], windows=(2,))
        team_a = rolled[rolled["team"] == "A"].sort_values("start_date")
        self.assertTrue(pd.isna(team_a.iloc[0]["points_for_l2"]))
        self.assertEqual(team_a.iloc[1]["points_for_l2"], 21)
        self.assertEqual(team_a.iloc[2]["points_for_l2"], 19)

    def test_asof_join_never_reads_future(self):
        observations = pd.DataFrame(
            {"team": ["A"], "prediction_time": pd.to_datetime(["2024-09-10"], utc=True)}
        )
        history = pd.DataFrame(
            {
                "team": ["A", "A"],
                "available_at": pd.to_datetime(["2024-09-01", "2024-09-12"], utc=True),
                "rating": [1.0, 99.0],
            }
        )
        joined = point_in_time_join(
            observations, history, by="team", observation_time="prediction_time",
            available_time="available_at", columns=["rating"],
        )
        self.assertEqual(joined.iloc[0]["rating"], 1.0)

    def test_game_observation_store_never_uses_future_value(self):
        games = pd.DataFrame(
            {
                "game_id": [1], "start_date": pd.to_datetime(["2026-09-10T17:00Z"]),
            }
        )
        observations = pd.DataFrame(
            {
                "entity_id": ["1", "1"], "entity_type": ["game", "game"],
                "feature_name": ["wind_speed", "wind_speed"], "value": [8, 99],
                "available_at": pd.to_datetime(["2026-09-09T12:00Z", "2026-09-11T12:00Z"]),
                "source_version": ["v1", "v1"],
            }
        )
        result = attach_game_observations(games, observations)
        self.assertEqual(result.iloc[0]["wind_speed"], 8)

    def test_quote_selection_is_asof_and_not_future(self):
        quotes = pd.DataFrame(
            {
                "game_id": [1, 1], "sportsbook": ["A", "A"],
                "market": ["total", "total"], "side": ["over", "over"],
                "captured_at": pd.to_datetime(["2026-09-09T12:00Z", "2026-09-11T12:00Z"]),
                "line": [52.5, 60.5], "odds": [-110, -110],
            }
        )
        quote = select_quote_as_of(
            quotes, game_id=1, market="total", side="over",
            as_of=pd.Timestamp("2026-09-10T12:00Z"),
        )
        self.assertEqual(quote["line"], 52.5)

    def test_walk_forward_seasons(self):
        frame = pd.DataFrame({"season": [2021, 2022, 2023, 2024], "x": range(4)})
        folds = list(walk_forward_season_splits(frame, min_train_seasons=2))
        self.assertEqual([fold.test_season for fold in folds], [2023, 2024])
        self.assertEqual(folds[0].train_seasons, (2021, 2022))

    def test_january_belongs_to_prior_cfb_season(self):
        self.assertEqual(current_cfb_season(date(2026, 1, 15)), 2025)
        self.assertEqual(rolling_season_window(at=date(2026, 1, 15))[-1], 2025)
        self.assertEqual(current_cfb_season(date(2026, 8, 15)), 2026)


class ContractAndFeatureTests(unittest.TestCase):
    def test_contract_detects_duplicates(self):
        games = pd.DataFrame(
            {
                "game_id": [1, 1], "season": [2026, 2026], "week": [1, 1],
                "home_team": ["A", "A"], "away_team": ["B", "B"],
                "start_date": ["2026-09-01", "2026-09-01"],
            }
        )
        self.assertFalse(validate_games(games).ok)
        self.assertFalse(validate_feature_matrix(games).ok)

    def test_safe_merge_blocks_row_explosion(self):
        left = pd.DataFrame({"id": [1, 2]})
        right = pd.DataFrame({"id": [1, 1], "value": [2, 3]})
        with self.assertRaises(pd.errors.MergeError):
            safe_merge(left, right, on="id", validate="many_to_one")

    def test_context_feature_library(self):
        frame = pd.DataFrame(
            {
                "week": [2], "season_type": ["regular"], "neutral_site": [False],
                "conference_game": [False], "home_conference": ["SEC"],
                "away_conference": ["Big Ten"], "home_rank": [5], "away_rank": [np.nan],
                "home_next_opponent_rank": [8], "rest_days_home": [14],
                "rest_days_away": [6], "temperature": [30], "wind_speed": [18],
                "precipitation": [0.2], "capacity": [100_000],
            }
        )
        result = build_context_features(frame)
        self.assertEqual(result.iloc[0]["home_trap_spot"], 1)
        self.assertEqual(result.iloc[0]["adverse_weather"], 1)
        self.assertEqual(result.iloc[0]["home_bye_week"], 1)
        self.assertGreaterEqual(len(result.columns) - len(frame.columns), 45)


class EvaluationAndRiskTests(unittest.TestCase):
    def test_probability_and_calibration_metrics(self):
        y = [0, 0, 1, 1]
        p = [0.1, 0.2, 0.8, 0.9]
        metrics = probability_metrics(y, p, baseline_probabilities=[0.5] * 4)
        self.assertLess(metrics["brier"], metrics["baseline_brier"])
        self.assertGreater(metrics["brier_skill"], 0)
        self.assertEqual(int(calibration_table(y, p, bins=2)["count"].sum()), 4)

    def test_baseline_metrics_use_identical_rows(self):
        probability = probability_metrics(
            [0, 1, 1], [0.1, 0.8, 0.1], baseline_probabilities=[0.4, 0.6, np.nan]
        )
        self.assertEqual(probability["baseline_n"], 2)
        self.assertLess(probability["model_brier_on_baseline_subset"], probability["baseline_brier"])

    def test_conformal_interval(self):
        lower, upper, width = conformal_interval([10, 20], [1, -2, 3, -4], alpha=0.25)
        self.assertGreater(width, 0)
        metrics = interval_metrics([10, 25], lower, upper)
        self.assertEqual(metrics["n"], 2)

    def test_risk_shrinks_and_caps(self):
        conservative = conservative_probability(
            0.62, market_probability=0.52, standard_error=0.02,
            shrinkage=0.25, uncertainty_z=1,
        )
        self.assertLess(conservative, 0.62)
        candidates = pd.DataFrame(
            {
                "game_id": [1, 1, 2], "team": ["A", "A", "C"],
                "model_probability": [0.62, 0.61, 0.60],
                "market_probability": [0.52, 0.52, 0.52],
                "odds": [-110, -110, -110],
            }
        )
        limits = RiskLimits(max_game_fraction=0.01, max_slate_fraction=0.02)
        sized = size_portfolio(candidates, bankroll=1000, limits=limits)
        self.assertLessEqual(sized.groupby("game_id")["stake_fraction"].sum().max(), 0.010001)
        self.assertLessEqual(sized["stake_fraction"].sum(), 0.020001)

    def test_release_gates_require_market_baseline_wins(self):
        metrics = {
            "win_model": {"n": 1000, "brier": 0.16, "baseline_brier": 0.18, "ece": 0.02},
            "spread_model": {
                "rmse": 16, "model_rmse_on_baseline_subset": 16, "baseline_rmse": 15,
            },
            "total_model": {
                "rmse": 14, "model_rmse_on_baseline_subset": 14, "baseline_rmse": 15,
            },
            "ats": {"positive_clv_rate": 0.55},
        }
        gates = {gate["name"]: gate for gate in evaluate_release_gates(metrics)}
        self.assertFalse(gates["spread_market_baseline"]["passed"])
        self.assertTrue(gates["total_market_baseline"]["passed"])


class JointScoreTests(unittest.TestCase):
    def test_coherent_probabilities(self):
        distribution = JointScoreDistribution(7, 55, 14, 16, 0.1)
        self.assertGreater(distribution.home_win_probability(), 0.5)
        self.assertAlmostEqual(distribution.home_cover_probability(-7), 0.5, places=6)
        self.assertAlmostEqual(distribution.over_probability(55), 0.5, places=6)
        self.assertAlmostEqual(
            distribution.mean_home_score + distribution.mean_away_score, 55
        )

    def test_residual_fit(self):
        margin_std, total_std, correlation = fit_residual_distribution(
            np.array([1, 2, 3, 4]), np.zeros(4),
            np.array([40, 42, 44, 46]), np.full(4, 43),
        )
        self.assertGreater(margin_std, 0)
        self.assertGreater(total_std, 0)
        self.assertTrue(-1 <= correlation <= 1)


if __name__ == "__main__":
    unittest.main()
