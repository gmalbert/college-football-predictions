"""Regression checks for the public Sports Picks Grid export contract."""
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import export_best_bets as exporter
from utils.betting import generate_spread_pick


class GridExportTests(unittest.TestCase):
    def test_spread_edge_uses_probability_instead_of_points(self):
        row = pd.Series({"game_id": 1, "start_date": "2026-10-24T20:00:00Z", "home_team": "Home", "away_team": "Away"})
        recommendation = generate_spread_pick("Home", "Away", 10, -3)
        bet = exporter._record(row, recommendation, line=-3, odds=-110, probability=.65)
        self.assertEqual(bet["confidence"], .65)
        self.assertEqual(bet["probability"], .65)
        self.assertEqual(bet["edge"], .1262)
        self.assertEqual(bet["source_edge"], 7)
        self.assertEqual(bet["source_edge_unit"], "points")
        self.assertEqual(bet["tier"], "Elite")

    def test_edge_tier_boundaries(self):
        for edge, tier in [(0, "Standard"), (.0099, "Standard"), (.01, "Good"), (.03, "Strong"), (.06, "Elite")]:
            with self.subTest(edge=edge):
                self.assertEqual(exporter._tier(edge), tier)

    def test_away_spread_exports_the_picked_sides_line_and_preserves_release_hold(self):
        frame = pd.DataFrame([{"game_id": 1, "start_date": f"{date.today()}T20:00:00Z", "home_team": "Home", "away_team": "Away", "predicted_spread": -10, "market_spread": -3}])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            feature = root / "data_files/features/feature_matrix.parquet"
            feature.parent.mkdir(parents=True)
            feature.touch()
            output = root / "best_bets_today.json"
            with patch.object(exporter, "ROOT", root), patch.object(exporter, "OUT_PATH", output), patch.object(exporter, "models_trained", return_value=True), patch.object(exporter, "load_metrics", return_value={"spread_model": {"residual_std": 10}, "release_decision": {"decision": "hold", "failed_gates": ["closing_line_value"]}}), patch.object(exporter.pd, "read_parquet", return_value=frame), patch.object(exporter, "predict_batch", side_effect=lambda f: f):
                exporter.main()
            data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(data["bets"][0]["pick"], "Take Away +3.0")
        self.assertEqual(data["bets"][0]["line"], 3)
        self.assertEqual(data["meta"]["tier_definition"], "edge-v1")
        self.assertEqual(data["meta"]["publication_mode"], "provisional_research")
        self.assertEqual(data["meta"]["release_status"], "hold")


if __name__ == "__main__":
    unittest.main()
