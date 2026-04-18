"""models/wpa_model.py

In-game Win Probability (WP) model, replicating cfbfastR's create_wpa_naive().

Requires EPA-enriched PBP data (run epa_model.compute_epa first).

Usage:
    from models.wpa_model import train_wp_model, predict_wp, compute_wpa
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

WP_FEATURES = [
    "ExpScoreDiff",
    "ExpScoreDiff_Time_Ratio",
    "half",
    "under_two",
    "pos_team_timeouts_rem",
    "def_pos_team_timeouts_rem",
    "pos_score_diff_start",
]

WP_MODEL_PATH = Path(__file__).resolve().parent.parent / "data_files" / "models" / "wp_model.joblib"


def _prepare_wp_features(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare WP model features from EPA-enriched PBP data."""
    df = pbp_df.copy()
    # Map period to half (periods 1-2 = first half, 3-4 = second half, OT = 2)
    df["half"] = np.where(df["period"].fillna(1).astype(int) <= 2, 1, 2)
    df["TimeSecsRem"] = (
        pd.to_numeric(df.get("clock_minutes", 0), errors="coerce") * 60 +
        pd.to_numeric(df.get("clock_seconds", 0), errors="coerce")
    )
    df["adj_TimeSecsRem"] = np.where(
        df["half"] == 1,
        1800 + df["TimeSecsRem"],
        df["TimeSecsRem"],
    )
    df["pos_score_diff_start"] = pd.to_numeric(
        df.get("score_diff"), errors="coerce"
    ).fillna(0)
    df["ep_before"] = pd.to_numeric(df.get("ep_before"), errors="coerce").fillna(0)
    df["ExpScoreDiff"] = df["pos_score_diff_start"] + df["ep_before"]
    df["ExpScoreDiff_Time_Ratio"] = df["ExpScoreDiff"] / (df["adj_TimeSecsRem"] + 1)
    df["under_two"] = (df["TimeSecsRem"] < 120).astype(int)
    df["pos_team_timeouts_rem"]     = pd.to_numeric(
        df.get("pos_team_timeouts_rem_before", 3), errors="coerce"
    ).fillna(3)
    df["def_pos_team_timeouts_rem"] = pd.to_numeric(
        df.get("def_pos_team_timeouts_rem_before", 3), errors="coerce"
    ).fillna(3)
    return df


def train_wp_model(pbp_df: pd.DataFrame) -> dict:
    """
    Train an in-game win probability model.

    Parameters
    ----------
    pbp_df : EPA-enriched PBP DataFrame with columns including:
             game_id, period, clock_minutes, clock_seconds, score_diff,
             ep_before, pos_team, winner (team name that won the game).

    Returns
    -------
    dict with keys 'model' and 'scaler'.
    """
    df = _prepare_wp_features(pbp_df)

    if "winner" not in df.columns or "pos_team" not in df.columns:
        logger.error("PBP must include 'pos_team' and 'winner' columns.")
        return {}

    df["pos_team_won"] = (df["pos_team"] == df["winner"]).astype(int)

    X = df[WP_FEATURES].dropna()
    y = df.loc[X.index, "pos_team_won"]

    if len(X) < 100:
        logger.warning("Too few plays to train WP model.")
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X_scaled, y)

    model_dict = {"model": model, "scaler": scaler}
    joblib.dump(model_dict, WP_MODEL_PATH)
    logger.info(f"WP model trained on {len(X):,} plays → {WP_MODEL_PATH}")
    return model_dict


def load_wp_model() -> dict:
    """Load the saved WP model, or return empty dict if not trained."""
    if WP_MODEL_PATH.exists():
        return joblib.load(WP_MODEL_PATH)
    return {}


def predict_wp(model_dict: dict, state_df: pd.DataFrame) -> np.ndarray:
    """Predict win probability for the possessing team."""
    df = _prepare_wp_features(state_df)
    X = df[WP_FEATURES].fillna(0).values
    X_scaled = model_dict["scaler"].transform(X)
    return model_dict["model"].predict_proba(X_scaled)[:, 1]


def compute_wpa(model_dict: dict, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute WPA (Win Probability Added) for each play.

    Returns pbp_df with new columns:
        wp_before, wp_after, wpa, home_wp
    """
    df = _prepare_wp_features(pbp_df)

    X = df[WP_FEATURES].fillna(0).values
    X_scaled = model_dict["scaler"].transform(X)
    df["wp_before"] = model_dict["model"].predict_proba(X_scaled)[:, 1]

    # wp_after = wp_before of the next play in the same game
    df["wp_after"] = df.groupby("game_id")["wp_before"].shift(-1)

    # Account for change of possession: if next play's pos_team differs, flip wp_after
    df["next_pos_team"] = df.groupby("game_id")["pos_team"].shift(-1)
    df["wpa"] = np.where(
        df["pos_team"] == df["next_pos_team"],
        df["wp_after"] - df["wp_before"],
        (1 - df["wp_after"]) - df["wp_before"],
    )

    # Convert to home-team perspective
    home_col = df.get("home") if "home" in df.columns else df.get("home_team")
    if home_col is not None:
        df["home_wp"] = np.where(
            df["pos_team"] == home_col,
            df["wp_before"],
            1 - df["wp_before"],
        )
    return df
