"""Uncertainty-aware Kelly sizing and correlated portfolio constraints."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.market import american_to_decimal, expected_value


@dataclass(frozen=True)
class RiskLimits:
    kelly_multiplier: float = 0.25
    max_bet_fraction: float = 0.01
    max_game_fraction: float = 0.015
    max_team_fraction: float = 0.03
    max_slate_fraction: float = 0.08
    min_expected_value: float = 0.01
    probability_shrinkage: float = 0.25
    uncertainty_z: float = 1.0


def kelly_fraction(win_probability: float, odds: float) -> float:
    probability = float(win_probability)
    if not 0 <= probability <= 1:
        raise ValueError("win_probability must be in [0, 1]")
    net = american_to_decimal(odds) - 1.0
    fraction = (net * probability - (1.0 - probability)) / net
    return max(0.0, float(fraction))


def conservative_probability(
    model_probability: float,
    *,
    market_probability: float = 0.5,
    standard_error: float = 0.0,
    shrinkage: float = 0.25,
    uncertainty_z: float = 1.0,
) -> float:
    """Shrink toward the no-vig market, then subtract an uncertainty haircut."""
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    blended = (1 - shrinkage) * float(model_probability) + shrinkage * float(market_probability)
    return float(np.clip(blended - uncertainty_z * max(0.0, standard_error), 0, 1))


def recommended_fraction(
    model_probability: float,
    odds: float,
    *,
    market_probability: float = 0.5,
    standard_error: float = 0.0,
    limits: RiskLimits = RiskLimits(),
) -> float:
    probability = conservative_probability(
        model_probability,
        market_probability=market_probability,
        standard_error=standard_error,
        shrinkage=limits.probability_shrinkage,
        uncertainty_z=limits.uncertainty_z,
    )
    if expected_value(probability, odds) < limits.min_expected_value:
        return 0.0
    sized = limits.kelly_multiplier * kelly_fraction(probability, odds)
    return min(sized, limits.max_bet_fraction)


def size_portfolio(
    candidates: pd.DataFrame,
    *,
    bankroll: float,
    limits: RiskLimits = RiskLimits(),
) -> pd.DataFrame:
    """Size candidates and enforce per-game, per-team, and slate exposure caps.

    Required columns are ``model_probability`` and ``odds``.  Optional
    ``market_probability``, ``probability_se``, ``game_id``, ``team``, and
    ``correlation_cluster`` columns enable progressively tighter controls.
    """
    required = {"model_probability", "odds"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise KeyError(f"candidates are missing columns: {missing}")
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")

    result = candidates.copy()
    market_source = (
        result["market_probability"]
        if "market_probability" in result.columns
        else pd.Series(0.5, index=result.index)
    )
    uncertainty_source = (
        result["probability_se"]
        if "probability_se" in result.columns
        else pd.Series(0.0, index=result.index)
    )
    market = pd.to_numeric(market_source, errors="coerce").fillna(0.5)
    uncertainty = pd.to_numeric(uncertainty_source, errors="coerce").fillna(0.0)
    result["sizing_probability"] = [
        conservative_probability(
            probability,
            market_probability=market_probability,
            standard_error=standard_error,
            shrinkage=limits.probability_shrinkage,
            uncertainty_z=limits.uncertainty_z,
        )
        for probability, market_probability, standard_error in zip(
            pd.to_numeric(result["model_probability"], errors="coerce"),
            market,
            uncertainty,
        )
    ]
    result["expected_value"] = [
        expected_value(probability, odds)
        for probability, odds in zip(result["sizing_probability"], result["odds"])
    ]
    result["raw_fraction"] = [
        min(
            limits.max_bet_fraction,
            limits.kelly_multiplier * kelly_fraction(probability, odds),
        )
        if ev >= limits.min_expected_value else 0.0
        for probability, odds, ev in zip(
            result["sizing_probability"], result["odds"], result["expected_value"]
        )
    ]

    # Correlated bets share the same risk budget. Scaling preserves relative
    # conviction while preventing a stack of bets on one underlying game/team.
    result["stake_fraction"] = result["raw_fraction"]
    for column, cap in (
        ("game_id", limits.max_game_fraction),
        ("team", limits.max_team_fraction),
        ("correlation_cluster", limits.max_game_fraction),
    ):
        if column not in result.columns:
            continue
        totals = result.groupby(column)["stake_fraction"].transform("sum")
        scale = np.minimum(1.0, cap / totals.replace(0, np.nan)).fillna(1.0)
        result["stake_fraction"] *= scale

    slate_total = float(result["stake_fraction"].sum())
    if slate_total > limits.max_slate_fraction:
        result["stake_fraction"] *= limits.max_slate_fraction / slate_total
    result["stake"] = result["stake_fraction"] * float(bankroll)
    result["rejected"] = result["stake_fraction"].eq(0)
    return result.sort_values(["stake_fraction", "expected_value"], ascending=False)
