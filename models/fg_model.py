"""models/fg_model.py

Field Goal Expected Points model using isotonic regression.
Replicates cfbfastR's GAM-based FG make probability by distance.

Usage:
    from models.fg_model import train_fg_model, predict_fg_prob, load_fg_model
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import joblib
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

FG_MODEL_PATH = Path(__file__).resolve().parent.parent / "data_files" / "models" / "fg_model.joblib"

# Typical snap distance from LOS + holder = ~8 yards to add to yards_to_goal
SNAP_HOLD_YARDS = 8


def train_fg_model(pbp_df: pd.DataFrame) -> IsotonicRegression:
    """
    Train FG make probability model based on distance.

    Parameters
    ----------
    pbp_df : PBP DataFrame with 'play_type' and 'yards_to_goal' columns.

    Returns
    -------
    Fitted IsotonicRegression model (make probability decreases with distance).
    """
    fg_mask = pbp_df["play_type"].str.contains("Field Goal", na=False)
    fg_plays = pbp_df[fg_mask].copy()

    if len(fg_plays) < 10:
        logger.warning("Too few FG plays to train model.")
        return IsotonicRegression(increasing=False, out_of_bounds="clip")

    fg_plays["fg_distance"] = (
        pd.to_numeric(fg_plays["yards_to_goal"], errors="coerce") + SNAP_HOLD_YARDS
    )
    fg_plays["fg_made"] = fg_plays["play_type"].str.contains("Good", na=False).astype(int)
    fg_plays = fg_plays.dropna(subset=["fg_distance"])

    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(fg_plays["fg_distance"].values, fg_plays["fg_made"].values)

    joblib.dump(model, FG_MODEL_PATH)
    made_pct = fg_plays["fg_made"].mean()
    logger.info(
        f"FG model trained on {len(fg_plays):,} attempts "
        f"(make rate: {made_pct:.1%}) → {FG_MODEL_PATH}"
    )
    return model


def load_fg_model() -> IsotonicRegression | None:
    """Load the saved FG model, or return None if not trained."""
    if FG_MODEL_PATH.exists():
        return joblib.load(FG_MODEL_PATH)
    return None


def predict_fg_prob(model: IsotonicRegression, distance: float | np.ndarray) -> float | np.ndarray:
    """
    Predict probability of making a field goal from given line-of-scrimmage distance.

    Parameters
    ----------
    model    : Fitted IsotonicRegression from train_fg_model().
    distance : Kick distance in yards (yards_to_goal + SNAP_HOLD_YARDS), or array thereof.

    Returns
    -------
    Make probability in [0, 1].
    """
    scalar = np.isscalar(distance)
    arr = np.atleast_1d(distance).astype(float)
    result = model.predict(arr)
    return float(result[0]) if scalar else result


def fg_ep_value(model: IsotonicRegression, yards_to_goal: float,
                miss_ep: float = -1.35) -> float:
    """
    Expected points for a FG attempt from a given yard line.

    EP_FG_attempt = P(make) * 3 + P(miss) * miss_ep

    miss_ep defaults to -1.35 (opponent gets ball at ~own 35, roughly -1.35 EP
    from kicking team perspective after a missed FG).
    """
    dist = float(yards_to_goal) + SNAP_HOLD_YARDS
    p_make = predict_fg_prob(model, dist)
    return p_make * 3.0 + (1 - p_make) * miss_ep
