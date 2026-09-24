"""Record model diagnostics into an artifact.

Two charts on the Model Performance page needed scikit-learn or XGBoost at
request time:

* the calibration curve called ``sklearn.calibration.calibration_curve``
* the feature-importance charts loaded every model artifact and then tried to
  pull importances out of them — which always failed, because the production
  spread and total models are ``MarketAnchoredRegressor`` wrappers with no
  intrinsic importance. The site was loading the training stack purely to
  print "not available".

Both are computed here instead, so the page renders whatever the pipeline
determined and the API needs neither library. It also means the page starts
showing real importances automatically if the models ever change to something
that exposes them.

Writes ``data_files/models/model_diagnostics.json``.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.feature_engine import SPREAD_FEATURES, TOTAL_FEATURES  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.model_artifacts import (  # noqa: E402
    MODEL_DIAGNOSTICS_PATH,
    load_metrics,
    models_trained,
)
from utils.models import load_models  # noqa: E402
from utils.storage import atomic_write_json, load_parquet  # noqa: E402

logger = get_logger(__name__)

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:  # noqa: BLE001
    xgb = None
    HAS_XGB = False

N_BINS = 10


def _calibration() -> dict:
    """Reliability-curve points for the out-of-sample win probabilities.

    Reimplements what sklearn's ``calibration_curve(strategy="uniform")`` does,
    so the pipeline — where sklearn is still available — owns the computation.
    """
    try:
        backtest = load_parquet("model_backtest", layer="features")
    except FileNotFoundError:
        return {"available": False, "reason": "model_backtest.parquet missing"}

    frame = backtest.dropna(subset=["win_prob_oos", "home_win"])
    if frame.empty:
        return {"available": False, "reason": "no out-of-sample win predictions"}

    probabilities = frame["win_prob_oos"].to_numpy(dtype=float)
    outcomes = frame["home_win"].to_numpy(dtype=float)

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_ids = np.searchsorted(edges[1:-1], probabilities)
    totals = np.bincount(bin_ids, minlength=len(edges))
    sums = np.bincount(bin_ids, weights=probabilities, minlength=len(edges))
    positives = np.bincount(bin_ids, weights=outcomes, minlength=len(edges))

    occupied = totals != 0
    return {
        "available": True,
        "n_bins": N_BINS,
        "n_samples": int(len(frame)),
        "mean_predicted": [float(v) for v in (sums[occupied] / totals[occupied])],
        "fraction_positive": [float(v) for v in (positives[occupied] / totals[occupied])],
    }


def _importance(model, features: list[str]) -> dict:
    """Return ``{"available": bool, ...}`` for one model."""
    if model is None:
        return {"available": False, "reason": "model artifact not loaded"}

    if HAS_XGB and isinstance(model, xgb.Booster):
        scores = model.get_score(importance_type="gain")
        return {
            "available": True,
            "kind": "xgboost_gain",
            "features": list(features),
            "importances": [
                float(scores.get(feature, scores.get(f"f{i}", 0.0)))
                for i, feature in enumerate(features)
            ],
        }

    try:
        coefficients = abs(model.named_steps["ridge"].coef_)
        used = list(model.named_steps["imputer"].get_feature_names_out(features))
        return {
            "available": True,
            "kind": "ridge_coefficients",
            "features": used,
            "importances": [float(value) for value in coefficients],
        }
    except Exception as exc:  # noqa: BLE001 - mirrors the page's former fallback
        return {
            "available": False,
            "reason": (
                f"{type(model).__name__} exposes no intrinsic feature importance "
                f"({type(exc).__name__})"
            ),
        }


def run() -> dict:
    if not models_trained():
        logger.warning("Models are not trained; skipping diagnostics export")
        return {"written": False, "reason": "models_not_trained"}

    models = load_models()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": load_metrics().get("model_version"),
        "calibration": _calibration(),
        "feature_importance": {
            "spread": _importance(models.get("spread"), SPREAD_FEATURES),
            "total": _importance(models.get("total"), TOTAL_FEATURES),
        },
    }
    atomic_write_json(MODEL_DIAGNOSTICS_PATH, payload)
    logger.info(
        "Wrote diagnostics (calibration=%s, spread importance=%s, total importance=%s)",
        payload["calibration"]["available"],
        payload["feature_importance"]["spread"]["available"],
        payload["feature_importance"]["total"]["available"],
    )
    return {
        "written": True,
        "calibration_available": payload["calibration"]["available"],
        "spread_importance_available": payload["feature_importance"]["spread"]["available"],
        "total_importance_available": payload["feature_importance"]["total"]["available"],
    }


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
