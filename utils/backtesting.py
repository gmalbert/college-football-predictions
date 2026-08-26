"""Price-aware bet selection and reproducible out-of-sample ledgers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.market import Market, Side, expected_value, settle_bet, spread_edge


@dataclass(frozen=True)
class BacktestConfig:
    min_expected_value: float = 0.02
    min_spread_edge: float = 1.5
    min_total_edge: float = 2.0
    flat_stake: float = 1.0
    default_odds: float = -110.0
    one_bet_per_game_market: bool = True


def build_spread_ledger(
    predictions: pd.DataFrame,
    *,
    config: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """Build and settle spread picks from strictly out-of-sample predictions.

    Required columns: game_id, home_score, away_score, home_spread,
    predicted_home_margin.  ``home_cover_probability`` and ``odds`` are used
    when present; without probability estimates, edge selection is points-only.
    """
    required = {
        "game_id", "home_score", "away_score", "home_spread", "predicted_home_margin",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise KeyError(f"spread predictions are missing columns: {missing}")
    if predictions["game_id"].duplicated().any():
        raise ValueError("spread backtest requires one prediction per game_id")

    rows = []
    for _, row in predictions.iterrows():
        edge = spread_edge(row["predicted_home_margin"], row["home_spread"])
        if abs(edge.edge_points) < config.min_spread_edge:
            continue
        side = edge.side
        odds_column = "home_odds" if side == Side.HOME else "away_odds"
        odds = float(row.get(odds_column, row.get("odds", config.default_odds)))
        probability = row.get("home_cover_probability", np.nan)
        if pd.notna(probability):
            probability = float(probability) if side == Side.HOME else 1.0 - float(probability)
            ev = expected_value(probability, odds)
            if ev < config.min_expected_value:
                continue
        else:
            ev = np.nan
        selected_line = float(row["home_spread"]) if side == Side.HOME else -float(row["home_spread"])
        settlement = settle_bet(
            market=Market.SPREAD,
            side=side,
            home_score=row["home_score"],
            away_score=row["away_score"],
            line=selected_line,
            odds=odds,
            stake=config.flat_stake,
        )
        rows.append(
            {
                "game_id": row["game_id"],
                "prediction_time": row.get("prediction_time"),
                "season": row.get("season"),
                "week": row.get("week"),
                "market": Market.SPREAD.value,
                "side": side.value,
                "line": selected_line,
                "odds": odds,
                "model_edge": abs(edge.edge_points),
                "model_probability": probability,
                "expected_value": ev,
                "stake": config.flat_stake,
                "result": settlement.result.value,
                "profit": settlement.profit,
            }
        )
    return pd.DataFrame(rows)


def build_total_ledger(
    predictions: pd.DataFrame,
    *,
    config: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    required = {
        "game_id", "home_score", "away_score", "total_line", "predicted_total",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise KeyError(f"total predictions are missing columns: {missing}")
    if predictions["game_id"].duplicated().any():
        raise ValueError("total backtest requires one prediction per game_id")
    rows = []
    for _, row in predictions.iterrows():
        edge = float(row["predicted_total"]) - float(row["total_line"])
        if abs(edge) < config.min_total_edge:
            continue
        side = Side.OVER if edge > 0 else Side.UNDER
        odds_column = "over_odds" if side == Side.OVER else "under_odds"
        odds = float(row.get(odds_column, row.get("odds", config.default_odds)))
        probability = row.get("over_probability", np.nan)
        if pd.notna(probability):
            probability = float(probability) if side == Side.OVER else 1.0 - float(probability)
            ev = expected_value(probability, odds)
            if ev < config.min_expected_value:
                continue
        else:
            ev = np.nan
        settlement = settle_bet(
            market=Market.TOTAL,
            side=side,
            home_score=row["home_score"],
            away_score=row["away_score"],
            line=float(row["total_line"]),
            odds=odds,
            stake=config.flat_stake,
        )
        rows.append(
            {
                "game_id": row["game_id"],
                "prediction_time": row.get("prediction_time"),
                "season": row.get("season"),
                "week": row.get("week"),
                "market": Market.TOTAL.value,
                "side": side.value,
                "line": float(row["total_line"]),
                "odds": odds,
                "model_edge": abs(edge),
                "model_probability": probability,
                "expected_value": ev,
                "stake": config.flat_stake,
                "result": settlement.result.value,
                "profit": settlement.profit,
            }
        )
    return pd.DataFrame(rows)


def assert_oos_ledger(
    ledger: pd.DataFrame,
    *,
    trained_through: str = "trained_through",
    prediction_time: str = "prediction_time",
) -> None:
    """Reject ledgers whose model training cutoff overlaps a prediction."""
    missing = [column for column in (trained_through, prediction_time) if column not in ledger.columns]
    if missing:
        raise KeyError(f"ledger is missing temporal audit columns: {missing}")
    trained = pd.to_datetime(ledger[trained_through], errors="coerce", utc=True)
    predicted = pd.to_datetime(ledger[prediction_time], errors="coerce", utc=True)
    invalid = trained.isna() | predicted.isna() | (trained >= predicted)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} ledger rows are not strictly out of sample")
