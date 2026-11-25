"""Benchmark market-anchored candidates using the production walk-forward folds."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.challenger_models import (  # noqa: E402
    cross_market_win_probabilities,
    walk_forward_market_calibration,
    walk_forward_market_residual,
    walk_forward_market_structural,
)
from utils.evaluation import probability_metrics, regression_metrics  # noqa: E402
from utils.feature_engine import WIN_FEATURES, SPREAD_FEATURES, TOTAL_FEATURES  # noqa: E402
from utils.market import remove_vig  # noqa: E402
from utils.storage import load_parquet  # noqa: E402
from utils.temporal import walk_forward_season_splits  # noqa: E402


def _market_home(frame: pd.DataFrame) -> np.ndarray:
    result = np.full(len(frame), np.nan)
    for index, (home, away) in enumerate(zip(frame.home_moneyline, frame.away_moneyline)):
        if pd.notna(home) and pd.notna(away) and float(home) != 0 and float(away) != 0:
            result[index] = remove_vig([home, away])[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-matrix", type=Path, help="Use a point-in-time feature Parquet instead of the repository artifact.")
    parser.add_argument("--cross-market-only", action="store_true", help="Benchmark only the cross-market moneyline challenger.")
    args = parser.parse_args()
    frame = pd.read_parquet(args.feature_matrix) if args.feature_matrix else load_parquet("feature_matrix", layer="features")
    frame = frame.drop_duplicates("game_id")
    priced = frame.dropna(subset=["home_win", "season"]).reset_index(drop=True)
    priced["market_home_prob"] = pd.to_numeric(priced["market_home_prob"], errors="coerce")
    for fold in walk_forward_season_splits(priced[["season", "start_date"]], min_train_seasons=2):
        train, test = priced.iloc[fold.train_index], priced.iloc[fold.test_index]
        forecast = cross_market_win_probabilities(train, test)
        print("win_cross_market", fold.test_season, probability_metrics(
            test["home_win"], forecast, baseline_probabilities=test["market_home_prob"]
        ))
    if args.cross_market_only:
        return

    frame = frame.dropna(subset=["home_win", "home_margin", "total_points", "season"]).copy()
    temporal = frame[["season", "start_date"]].reset_index(drop=True)
    market_home = _market_home(frame)
    win_oof, win_folds = walk_forward_market_calibration(
        market_home, frame.home_win.to_numpy(float), temporal
    )
    print("win_market_calibrated", probability_metrics(
        frame.home_win, win_oof, baseline_probabilities=market_home
    ), win_folds)

    win_feature_sets = {
        "elo": ["elo_diff", "elo_home_win_prob"],
        "core": [
            "elo_diff", "elo_home_win_prob", "recruiting_diff",
            "recruiting_rank_diff", "returning_ppa_diff", "coach_tenure_diff",
            "portal_net_diff", "rest_advantage", "margin_diff_l5",
            "turnover_margin_l5",
        ],
        "all": WIN_FEATURES,
    }
    for name, columns in win_feature_sets.items():
        for c in (0.01, 0.05, 0.2):
            prediction, folds = walk_forward_market_structural(
                frame[columns], market_home, frame.home_win.to_numpy(float),
                temporal, c=c,
            )
            print("win_market_structural", name, c, probability_metrics(
                frame.home_win, prediction, baseline_probabilities=market_home
            ), folds)

    for kind in ("ridge", "hist", "extra_trees"):
        spread_market = -pd.to_numeric(frame.market_spread, errors="coerce").to_numpy(float)
        spread_oof, spread_folds = walk_forward_market_residual(
            frame[SPREAD_FEATURES], frame.home_margin.to_numpy(float), spread_market,
            temporal, kind=kind,
        )
        print("spread", kind, regression_metrics(
            frame.home_margin, spread_oof, baseline_predictions=spread_market
        ), spread_folds)

        total_market = pd.to_numeric(frame.market_total, errors="coerce").to_numpy(float)
        total_oof, total_folds = walk_forward_market_residual(
            frame[TOTAL_FEATURES], frame.total_points.to_numpy(float), total_market,
            temporal, kind=kind,
        )
        print("total", kind, regression_metrics(
            frame.total_points, total_oof, baseline_predictions=total_market
        ), total_folds)


if __name__ == "__main__":
    main()
