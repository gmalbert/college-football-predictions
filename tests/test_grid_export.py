"""Regression checks for the public Sports Picks Grid export contract."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import export_best_bets


class TestPublicBestBetsExport(unittest.TestCase):
    def test_hold_overwrites_previous_public_picks(self):
        decision = {"decision": "hold", "failed_gates": ["closing_line_value"]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "best_bets_today.json"
            output.write_text('{"bets": [{"tier": "Elite"}]}', encoding="utf-8")
            with (
                patch.object(export_best_bets, "OUT_PATH", output),
                patch.object(export_best_bets, "load_metrics", return_value={"release_decision": decision}),
                patch.object(export_best_bets, "predict_batch") as predict,
            ):
                export_best_bets.main()
                predict.assert_not_called()
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["bets"], [])
        self.assertEqual(payload["meta"]["release_status"], "hold")
        self.assertEqual(payload["meta"]["release_failed_gates"], ["closing_line_value"])

    def test_promoted_export_keeps_spread_price_edge_and_withholds_totals(self):
        now = pd.Timestamp("2026-11-25T20:00:00Z")
        game = {
            "game_id": 401934100,
            "home_team": "Tennessee Tech",
            "away_team": "Dayton",
            "start_date": now + pd.Timedelta(days=3),
            "market_spread": -21.5,
            "predicted_spread": 26.1681,
            "market_total": 42.5,
            "predicted_total": 51.0,
        }
        quotes = pd.DataFrame([
            {
                "game_id": game["game_id"], "market": market, "side": side,
                "line": line, "odds": odds, "sportsbook": "FanDuel",
                "source": "therundown", "captured_at": now - pd.Timedelta(minutes=10),
            }
            for market, side, line, odds in (
                ("spread", "home", -21.5, -104),
                ("spread", "away", 21.5, -122),
                ("total", "over", 42.5, -110),
                ("total", "under", 42.5, -110),
            )
        ])
        metrics = {
            "release_decision": {"decision": "promote"},
            "spread_model": {"residual_std": 18},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "data_files" / "features" / "feature_matrix.parquet"
            features.parent.mkdir(parents=True)
            pd.DataFrame([game]).to_parquet(features)
            snapshot = root / "line_snapshots.parquet"
            quotes.to_parquet(snapshot)
            output = root / "best_bets_today.json"
            with (
                patch.object(export_best_bets, "ROOT", root),
                patch.object(export_best_bets, "OUT_PATH", output),
                patch.object(export_best_bets, "SNAPSHOT_PATH", snapshot),
                patch.object(export_best_bets.pd.Timestamp, "now", return_value=now),
                patch.object(export_best_bets, "load_metrics", return_value=metrics),
                patch.object(export_best_bets, "models_trained", return_value=True),
                patch.object(export_best_bets, "predict_batch", side_effect=lambda frame: frame),
            ):
                export_best_bets.main()
                payload = json.loads(output.read_text(encoding="utf-8"))
                # An opposing quote from a different provider cannot complete the pair.
                quotes.loc[quotes["side"].eq("away"), "source"] = "odds_api"
                quotes.to_parquet(snapshot)
                export_best_bets.main()
                unpaired_payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(unpaired_payload["bets"], [])
        self.assertEqual(payload["meta"]["publication_mode"], "validated")
        self.assertEqual(payload["meta"]["totals_publication"], "research_only")
        bet, = payload["bets"]
        self.assertEqual(bet["bet_type"], "Spread")
        self.assertEqual(bet["line"], -21.5)
        self.assertEqual(bet["market_quote_count"], 1)
        self.assertEqual(bet["opposing_odds"], -122)
        self.assertAlmostEqual(bet["price_implied_probability"], 104 / 204, places=6)
        self.assertAlmostEqual(bet["edge"], bet["probability"] - 104 / 204, places=4)
        self.assertGreater(bet["market_edge"], bet["edge"])

    def test_edge_tier_boundaries(self):
        for edge, tier in [(0, "Standard"), (.0099, "Standard"), (.01, "Good"), (.03, "Strong"), (.06, "Elite")]:
            with self.subTest(edge=edge):
                self.assertEqual(export_best_bets._tier(edge), tier)

if __name__ == "__main__":
    unittest.main()
