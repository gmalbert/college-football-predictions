"""Read-only access to the pipeline's model artifacts.

Everything here works from files on disk — the metrics JSON, the model
manifests, ``model_backtest.parquet`` and ``upcoming_predictions.parquet``.
Nothing in this module imports scikit-learn or XGBoost, which is deliberate:
the dashboard serves forecasts that the weekly workflow already computed, so
it never needs the training stack.

``utils/models.py`` remains the training and inference module. It re-exports
everything below so existing callers keep working.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.storage import MODELS_DIR, load_parquet

__all__ = [
    "WIN_MODEL_PATH",
    "SPREAD_MODEL_PATH",
    "TOTAL_MODEL_PATH",
    "TOTAL_COVER_MODEL_PATH",
    "METRICS_PATH",
    "MODEL_VERSION",
    "PREDICTION_COLUMNS",
    "UPCOMING_PREDICTIONS_ARTIFACT",
    "MODEL_DIAGNOSTICS_PATH",
    "completed_mask",
    "load_upcoming_predictions",
    "attach_predictions",
    "predict_for_display",
    "load_metrics",
    "models_trained",
    "load_model_diagnostics",
]

WIN_MODEL_PATH = MODELS_DIR / "win_prob_model.joblib"
SPREAD_MODEL_PATH = MODELS_DIR / "spread_model.joblib"
TOTAL_MODEL_PATH = MODELS_DIR / "total_model.joblib"
TOTAL_COVER_MODEL_PATH = MODELS_DIR / "total_cover_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
MODEL_VERSION = "2.2.0"

# Written by scripts/export_model_diagnostics.py in the weekly workflow, so the
# Model Performance page can render its calibration and importance charts
# without loading scikit-learn or XGBoost.
MODEL_DIAGNOSTICS_PATH = MODELS_DIR / "model_diagnostics.json"

PREDICTION_COLUMNS = (
    "win_prob",
    "predicted_spread",
    "predicted_total",
    "total_over_prob",
)

UPCOMING_PREDICTIONS_ARTIFACT = "upcoming_predictions"


def completed_mask(frame: pd.DataFrame) -> pd.Series:
    """True where a game already has a result."""
    if "home_margin" in frame.columns:
        return pd.to_numeric(frame["home_margin"], errors="coerce").notna()
    if {"home_score", "away_score"}.issubset(frame.columns):
        return frame["home_score"].notna() & frame["away_score"].notna()
    return pd.Series(False, index=frame.index)


def load_upcoming_predictions() -> pd.DataFrame | None:
    """Load the pipeline-materialised predictions for unplayed games.

    ``scripts/export_upcoming_predictions.py`` writes this in the weekly
    workflow. Reading it is what lets the site serve forecasts without loading
    scikit-learn or XGBoost, or scoring a single row at request time.
    """
    try:
        frame = load_parquet(UPCOMING_PREDICTIONS_ARTIFACT, layer="features")
    except FileNotFoundError:
        return None
    if frame.empty or "game_id" not in frame.columns:
        return None
    frame = frame.copy()
    frame["game_id"] = pd.to_numeric(frame["game_id"], errors="coerce")
    return frame.dropna(subset=["game_id"]).drop_duplicates("game_id").set_index("game_id")


def _attach_by_game_id(
    result: pd.DataFrame, lookup: pd.DataFrame, mask: pd.Series | None = None
) -> None:
    """Map prediction columns onto ``result`` by ``game_id``, in place."""
    if "game_id" not in result.columns:
        return
    ids = pd.to_numeric(result["game_id"], errors="coerce")
    for column in PREDICTION_COLUMNS:
        if column not in lookup.columns:
            continue
        mapped = ids.map(lookup[column]).to_numpy()
        if mask is None:
            result[column] = mapped
        else:
            result.loc[mask, column] = mapped[mask.to_numpy()]


def attach_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with pipeline predictions attached.

    Drop-in replacement for the old ``predict_batch`` call on the home page:
    the values are the same, they were just computed in the workflow instead of
    per request.
    """
    result = frame.copy()
    for column in PREDICTION_COLUMNS:
        result[column] = np.nan
    lookup = load_upcoming_predictions()
    if lookup is not None and not result.empty:
        _attach_by_game_id(result, lookup)
    return result


def predict_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return honest predictions for mixed historical/upcoming UI slices.

    Completed games receive their saved season walk-forward predictions;
    unplayed games receive the pipeline-materialised full-history forecast.
    Nothing is inferred here — both sets are read from artifacts produced by
    the weekly workflow. ``prediction_scope`` makes the distinction visible.
    """
    result = df.copy()
    for column in PREDICTION_COLUMNS:
        result[column] = np.nan
    result["prediction_scope"] = "unavailable"

    completed = completed_mask(result)
    upcoming = ~completed

    upcoming_lookup = load_upcoming_predictions()
    if upcoming.any() and upcoming_lookup is not None:
        _attach_by_game_id(result, upcoming_lookup, mask=upcoming)
        result.loc[upcoming, "prediction_scope"] = "future_full_fit"

    if completed.any() and "game_id" in result.columns:
        try:
            backtest = load_parquet("model_backtest", layer="features")
        except FileNotFoundError:
            backtest = pd.DataFrame()
        if not backtest.empty and "game_id" in backtest.columns:
            oos_columns = {
                "win_prob_oos": "win_prob",
                "predicted_spread_oos": "predicted_spread",
                "predicted_total_oos": "predicted_total",
                "total_over_prob_oos": "total_over_prob",
            }
            available = ["game_id", *[c for c in oos_columns if c in backtest.columns]]
            lookup = backtest[available].drop_duplicates("game_id").set_index("game_id")
            for source, target in oos_columns.items():
                if source in lookup.columns:
                    mapped = result.loc[completed, "game_id"].map(lookup[source])
                    result.loc[completed, target] = mapped.to_numpy()
            has_oos = result.loc[completed, list(PREDICTION_COLUMNS)].notna().any(axis=1)
            result.loc[has_oos.index[has_oos], "prediction_scope"] = "walk_forward_oos"
    return result


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    return {}


def models_trained() -> bool:
    return all(
        p.exists() for p in [
            WIN_MODEL_PATH, SPREAD_MODEL_PATH, TOTAL_MODEL_PATH,
            TOTAL_COVER_MODEL_PATH,
        ]
    )


def load_model_diagnostics() -> dict:
    """The calibration and importance records the workflow exported."""
    if not MODEL_DIAGNOSTICS_PATH.exists():
        return {}
    try:
        return json.loads(MODEL_DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
