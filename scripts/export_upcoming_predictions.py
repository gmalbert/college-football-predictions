"""Materialise model forecasts for unplayed games into an artifact.

The dashboard must not run model inference. Everything that needs a model is
computed here, in the weekly workflow, and read back by the site — the same
pattern ``export_best_bets.py`` and ``export_shadow_totals.py`` already use.

Writes ``data_files/features/upcoming_predictions.parquet``:
``game_id, win_prob, predicted_spread, predicted_total, total_over_prob``

``utils.models.predict_for_display`` and ``attach_predictions`` read it, which
is what lets the API serve forecasts without loading scikit-learn or XGBoost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logger import get_logger  # noqa: E402
from utils.models import (  # noqa: E402
    PREDICTION_COLUMNS,
    UPCOMING_PREDICTIONS_ARTIFACT,
    completed_mask,
    models_trained,
    predict_batch,
)
from utils.storage import FEATURES_DIR, load_parquet, save_parquet  # noqa: E402

logger = get_logger(__name__)


def run() -> dict:
    """Write the upcoming-prediction artifact. Returns a small summary."""
    if not models_trained():
        logger.warning("Models are not trained; skipping upcoming-prediction export")
        return {"written": 0, "reason": "models_not_trained"}

    try:
        features = load_parquet("feature_matrix", layer="features")
    except FileNotFoundError:
        logger.warning("feature_matrix.parquet missing; skipping export")
        return {"written": 0, "reason": "no_feature_matrix"}

    upcoming = features[~completed_mask(features)].copy()
    if upcoming.empty:
        logger.info("No unplayed games in the feature matrix; writing empty artifact")
        empty = pd.DataFrame(columns=["game_id", *PREDICTION_COLUMNS])
        save_parquet(empty, UPCOMING_PREDICTIONS_ARTIFACT, layer="features")
        return {"written": 0, "reason": "no_upcoming_games"}

    scored = predict_batch(upcoming)
    columns = ["game_id", *[c for c in PREDICTION_COLUMNS if c in scored.columns]]
    artifact = scored[columns].copy()
    artifact["game_id"] = pd.to_numeric(artifact["game_id"], errors="coerce")
    artifact = artifact.dropna(subset=["game_id"]).drop_duplicates("game_id")
    artifact["game_id"] = artifact["game_id"].astype("int64")

    path = save_parquet(artifact, UPCOMING_PREDICTIONS_ARTIFACT, layer="features")
    logger.info(f"Wrote {len(artifact):,} upcoming predictions to {path.name}")
    return {
        "written": int(len(artifact)),
        "columns": columns,
        "path": str(path.relative_to(FEATURES_DIR.parent)),
    }


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
