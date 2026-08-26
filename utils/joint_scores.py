"""Coherent joint margin/total distribution for win, spread, and total markets."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class JointScoreDistribution:
    """Bivariate-normal approximation over home margin and game total."""

    mean_margin: float
    mean_total: float
    margin_std: float
    total_std: float
    correlation: float = 0.0

    def __post_init__(self) -> None:
        if self.margin_std <= 0 or self.total_std <= 0:
            raise ValueError("standard deviations must be positive")
        if not -0.99 <= self.correlation <= 0.99:
            raise ValueError("correlation must be between -0.99 and 0.99")

    @property
    def mean_home_score(self) -> float:
        return (self.mean_total + self.mean_margin) / 2.0

    @property
    def mean_away_score(self) -> float:
        return (self.mean_total - self.mean_margin) / 2.0

    def home_win_probability(self) -> float:
        z = (0.0 - self.mean_margin) / self.margin_std
        return 1.0 - NormalDist().cdf(z)

    def home_cover_probability(self, home_spread: float) -> float:
        threshold = -float(home_spread)
        z = (threshold - self.mean_margin) / self.margin_std
        return 1.0 - NormalDist().cdf(z)

    def over_probability(self, total_line: float) -> float:
        z = (float(total_line) - self.mean_total) / self.total_std
        return 1.0 - NormalDist().cdf(z)

    def margin_interval(self, confidence: float = 0.90) -> tuple[float, float]:
        z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
        return self.mean_margin - z * self.margin_std, self.mean_margin + z * self.margin_std

    def total_interval(self, confidence: float = 0.90) -> tuple[float, float]:
        z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
        return self.mean_total - z * self.total_std, self.mean_total + z * self.total_std

    def simulate(self, samples: int = 10_000, seed: int = 42) -> dict[str, np.ndarray]:
        if samples <= 0:
            raise ValueError("samples must be positive")
        covariance = np.array(
            [
                [self.margin_std**2, self.correlation * self.margin_std * self.total_std],
                [self.correlation * self.margin_std * self.total_std, self.total_std**2],
            ]
        )
        rng = np.random.default_rng(seed)
        margin, total = rng.multivariate_normal(
            [self.mean_margin, self.mean_total], covariance, size=samples
        ).T
        home = np.maximum(0.0, (total + margin) / 2.0)
        away = np.maximum(0.0, (total - margin) / 2.0)
        return {
            "margin": home - away,
            "total": home + away,
            "home_score": home,
            "away_score": away,
        }


def fit_residual_distribution(
    actual_margin: np.ndarray,
    predicted_margin: np.ndarray,
    actual_total: np.ndarray,
    predicted_total: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate out-of-sample residual scale/correlation for joint forecasts."""
    margin_residual = np.asarray(actual_margin, dtype=float) - np.asarray(
        predicted_margin, dtype=float
    )
    total_residual = np.asarray(actual_total, dtype=float) - np.asarray(
        predicted_total, dtype=float
    )
    valid = np.isfinite(margin_residual) & np.isfinite(total_residual)
    if valid.sum() < 3:
        raise ValueError("at least three paired residuals are required")
    margin_residual = margin_residual[valid]
    total_residual = total_residual[valid]
    return (
        float(np.std(margin_residual, ddof=1)),
        float(np.std(total_residual, ddof=1)),
        float(np.corrcoef(margin_residual, total_residual)[0, 1]),
    )
