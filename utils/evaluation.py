"""Out-of-sample probability, regression, interval, and betting evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class ReleaseGate:
    name: str
    passed: bool
    actual: float | int | str | None
    threshold: float | int | str
    reason: str


def calibration_table(
    y_true: Sequence[float],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], np.clip(p[valid], 0, 1)
    if not len(y):
        return pd.DataFrame(columns=["bin", "count", "mean_probability", "event_rate", "gap"])
    edges = np.linspace(0, 1, bins + 1)
    labels = np.minimum(np.digitize(p, edges[1:-1], right=True), bins - 1)
    rows = []
    for label in range(bins):
        mask = labels == label
        if not mask.any():
            continue
        mean_probability = float(p[mask].mean())
        event_rate = float(y[mask].mean())
        rows.append(
            {
                "bin": label,
                "lower": float(edges[label]),
                "upper": float(edges[label + 1]),
                "count": int(mask.sum()),
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "gap": abs(mean_probability - event_rate),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: Sequence[float],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    table = calibration_table(y_true, probabilities, bins=bins)
    if table.empty:
        return float("nan")
    return float(np.average(table["gap"], weights=table["count"]))


def probability_metrics(
    y_true: Sequence[float],
    probabilities: Sequence[float],
    *,
    baseline_probabilities: Sequence[float] | None = None,
    bins: int = 10,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], np.clip(p[valid], 1e-6, 1 - 1e-6)
    if not len(y):
        return {"n": 0, "brier": float("nan"), "log_loss": float("nan"), "ece": float("nan")}
    brier = float(brier_score_loss(y, p))
    result: dict[str, float | int] = {
        "n": int(len(y)),
        "brier": brier,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p, bins=bins),
        "accuracy": float(((p >= 0.5) == y).mean()),
    }
    if baseline_probabilities is not None:
        baseline = np.asarray(baseline_probabilities, dtype=float)[valid]
        base_valid = np.isfinite(baseline)
        if base_valid.any():
            baseline_brier = float(
                brier_score_loss(y[base_valid], np.clip(baseline[base_valid], 0, 1))
            )
            comparable_brier = float(brier_score_loss(y[base_valid], p[base_valid]))
            result["baseline_n"] = int(base_valid.sum())
            result["baseline_brier"] = baseline_brier
            result["model_brier_on_baseline_subset"] = comparable_brier
            result["brier_skill"] = (
                1.0 - comparable_brier / baseline_brier
                if baseline_brier > 0 else float("nan")
            )
    return result


def regression_metrics(
    y_true: Sequence[float],
    predictions: Sequence[float],
    *,
    baseline_predictions: Sequence[float] | None = None,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predictions, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        return {"n": 0, "rmse": float("nan"), "mae": float("nan")}
    result: dict[str, float | int] = {
        "n": int(len(y)),
        "rmse": float(sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "bias": float((p - y).mean()),
        "residual_std": float(np.std(y - p, ddof=1)) if len(y) > 1 else float("nan"),
    }
    if baseline_predictions is not None:
        baseline = np.asarray(baseline_predictions, dtype=float)[valid]
        base_valid = np.isfinite(baseline)
        if base_valid.any():
            comparable_rmse = float(
                sqrt(mean_squared_error(y[base_valid], p[base_valid]))
            )
            comparable_mae = float(mean_absolute_error(y[base_valid], p[base_valid]))
            result["baseline_n"] = int(base_valid.sum())
            result["baseline_rmse"] = float(
                sqrt(mean_squared_error(y[base_valid], baseline[base_valid]))
            )
            result["baseline_mae"] = float(
                mean_absolute_error(y[base_valid], baseline[base_valid])
            )
            result["model_rmse_on_baseline_subset"] = comparable_rmse
            result["model_mae_on_baseline_subset"] = comparable_mae
    return result


def conformal_interval(
    predictions: Sequence[float],
    calibration_residuals: Sequence[float],
    *,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Symmetric split-conformal interval using only calibration residuals."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    residuals = np.abs(np.asarray(calibration_residuals, dtype=float))
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        raise ValueError("at least one finite calibration residual is required")
    rank = min(1.0, np.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals))
    quantile = float(np.quantile(residuals, rank, method="higher"))
    point = np.asarray(predictions, dtype=float)
    return point - quantile, point + quantile, quantile


def interval_metrics(
    y_true: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[valid], lo[valid], hi[valid]
    return {
        "n": int(len(y)),
        "coverage": float(((y >= lo) & (y <= hi)).mean()) if len(y) else float("nan"),
        "mean_width": float((hi - lo).mean()) if len(y) else float("nan"),
    }


def max_drawdown(bankroll: Sequence[float]) -> float:
    values = np.asarray(bankroll, dtype=float)
    if not len(values):
        return float("nan")
    peaks = np.maximum.accumulate(values)
    drawdowns = np.divide(values - peaks, peaks, out=np.zeros_like(values), where=peaks != 0)
    return float(drawdowns.min())


def betting_metrics(bets: pd.DataFrame) -> dict[str, float | int]:
    """Summarize a settled bet ledger with pushes and voids excluded from hit rate."""
    required = {"result", "stake", "profit"}
    missing = sorted(required.difference(bets.columns))
    if missing:
        raise KeyError(f"bet ledger is missing columns: {missing}")
    settled = bets[bets["result"].isin(["W", "L", "P"])].copy()
    decisions = settled[settled["result"].isin(["W", "L"])]
    stake = float(pd.to_numeric(settled["stake"], errors="coerce").sum())
    profit = float(pd.to_numeric(settled["profit"], errors="coerce").sum())
    bankroll = 1.0 + pd.to_numeric(settled["profit"], errors="coerce").fillna(0).cumsum()
    result: dict[str, float | int] = {
        "bets": int(len(settled)),
        "wins": int((decisions["result"] == "W").sum()),
        "losses": int((decisions["result"] == "L").sum()),
        "pushes": int((settled["result"] == "P").sum()),
        "hit_rate": float((decisions["result"] == "W").mean()) if len(decisions) else float("nan"),
        "staked": stake,
        "profit": profit,
        "roi": profit / stake if stake else float("nan"),
        "max_drawdown": max_drawdown(bankroll),
    }
    if "clv" in settled.columns:
        result["mean_clv"] = float(pd.to_numeric(settled["clv"], errors="coerce").mean())
        result["positive_clv_rate"] = float(
            (pd.to_numeric(settled["clv"], errors="coerce") > 0).mean()
        )
    return result


def cluster_bootstrap_interval(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    cluster: str = "season_week",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap whole temporal clusters instead of pretending bets are iid."""
    if cluster not in frame.columns:
        raise KeyError(f"missing cluster column: {cluster}")
    groups = [group for _, group in frame.groupby(cluster, sort=False)]
    if len(groups) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[index] for index in chosen], ignore_index=True)
        estimates.append(float(metric(sample)))
    tail = (1 - confidence) / 2
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail))


def evaluate_release_gates(
    metrics: dict,
    *,
    min_oos_games: int = 500,
    max_brier: float = 0.25,
    max_ece: float = 0.05,
    min_positive_clv_rate: float = 0.50,
) -> list[dict]:
    probability = metrics.get("win_model", {})
    spread = metrics.get("spread_model", {})
    total = metrics.get("total_model", {})
    betting = metrics.get("betting", metrics.get("ats", {}))
    gates = [
        ReleaseGate(
            "out_of_sample_games",
            int(probability.get("n", probability.get("n_samples", 0))) >= min_oos_games,
            int(probability.get("n", probability.get("n_samples", 0))),
            min_oos_games,
            "Require an adequately sized untouched walk-forward sample.",
        ),
        ReleaseGate(
            "brier",
            float(probability.get("brier", np.inf)) <= max_brier,
            probability.get("brier"),
            max_brier,
            "Probability quality must beat a coin-flip Brier score.",
        ),
        ReleaseGate(
            "win_market_baseline",
            float(probability.get("model_brier_on_baseline_subset", np.inf))
            < float(probability.get("baseline_brier", -np.inf)),
            probability.get("brier_skill"),
            "> 0 Brier skill",
            "Win probabilities must improve on the vig-removed moneyline market.",
        ),
        ReleaseGate(
            "calibration",
            float(probability.get("ece", np.inf)) <= max_ece,
            probability.get("ece"),
            max_ece,
            "Calibration error controls overconfident bet sizing.",
        ),
        ReleaseGate(
            "spread_market_baseline",
            float(spread.get("model_rmse_on_baseline_subset", np.inf))
            < float(spread.get("baseline_rmse", -np.inf)),
            spread.get("model_rmse_on_baseline_subset"),
            f"< market RMSE {spread.get('baseline_rmse')}",
            "A spread model is not releasable when the market predicts margin better.",
        ),
        ReleaseGate(
            "total_market_baseline",
            float(total.get("model_rmse_on_baseline_subset", np.inf))
            < float(total.get("baseline_rmse", -np.inf)),
            total.get("model_rmse_on_baseline_subset"),
            f"< market RMSE {total.get('baseline_rmse')}",
            "A total model is not releasable when the market predicts totals better.",
        ),
        ReleaseGate(
            "closing_line_value",
            float(betting.get("positive_clv_rate", -np.inf)) >= min_positive_clv_rate,
            betting.get("positive_clv_rate"),
            min_positive_clv_rate,
            "A live strategy should beat later market prices more often than not.",
        ),
    ]
    return [asdict(gate) for gate in gates]
