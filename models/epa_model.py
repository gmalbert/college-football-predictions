"""models/epa_model.py

Custom Expected Points (EP) model trained on play-by-play data.
Replicates cfbfastR's create_epa() as a Python multinomial logistic regression.

Usage:
    from models.epa_model import train_ep_model, compute_epa, predict_ep

The model predicts the next scoring event (7 outcomes) from pre-play game state,
then computes EPA = EP_after - EP_before.
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

SCORING_WEIGHTS = {
    "No_Score": 0, "FG": 3, "Opp_FG": -3,
    "Opp_Safety": -2, "Opp_TD": -7, "Safety": 2, "TD": 7,
}
CLASSES = ["No_Score", "FG", "Opp_FG", "Opp_Safety", "Opp_TD", "Safety", "TD"]

EP_FEATURES = [
    "TimeSecsRem", "down_2", "down_3", "down_4",
    "distance", "yards_to_goal", "log_ydstogo",
    "under_two", "goal_to_go", "pos_score_diff_start",
]

EP_MODEL_PATH = Path(__file__).resolve().parent.parent / "data_files" / "models" / "ep_model.joblib"


def _classify_score(play: dict | pd.Series) -> str:
    """Classify a scoring play into one of 7 next-score types."""
    pt = str(play.get("play_type", ""))
    home_team = play.get("home")
    pos_team  = play.get("pos_team") or play.get("offense")
    if "Touchdown" in pt:
        return "TD" if pos_team == home_team else "Opp_TD"
    if "Field Goal Good" in pt:
        return "FG" if pos_team == home_team else "Opp_FG"
    if "Safety" in pt:
        return "Safety" if pos_team != home_team else "Opp_Safety"
    return "No_Score"


def _label_next_score(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Label each play with the next scoring event in its half (no data leakage)."""
    df = pbp_df.sort_values(["game_id", "play_number"]).copy()
    df["next_score_type"] = "No_Score"

    for game_id in df["game_id"].unique():
        game = df[df["game_id"] == game_id]
        for half in [1, 2]:
            half_mask  = game["period"].isin([half * 2 - 1, half * 2])
            half_plays = game[half_mask].index.tolist()
            last_score = "No_Score"
            for idx in reversed(half_plays):
                row = df.loc[idx]
                if row.get("scoring_play", False):
                    last_score = _classify_score(row)
                df.loc[idx, "next_score_type"] = last_score
    return df


def _prepare_ep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build EP feature columns from raw PBP rows."""
    df = df.copy()
    df["down"] = pd.to_numeric(df.get("down"), errors="coerce")
    df = df[df["down"].isin([1, 2, 3, 4])].copy()

    df["distance"]          = pd.to_numeric(df.get("distance"), errors="coerce")
    df["yards_to_goal"]     = pd.to_numeric(df.get("yards_to_goal"), errors="coerce")
    df["TimeSecsRem"]       = pd.to_numeric(df.get("clock_minutes", 0), errors="coerce") * 60 + \
                              pd.to_numeric(df.get("clock_seconds", 0), errors="coerce")
    df["pos_score_diff_start"] = pd.to_numeric(df.get("score_diff"), errors="coerce").fillna(0)
    df["log_ydstogo"]       = np.log(df["distance"].clip(lower=1))
    df["under_two"]         = (df["TimeSecsRem"] < 120).astype(int)
    df["goal_to_go"]        = (df["yards_to_goal"] == df["distance"]).astype(int)

    # One-hot encode down (reference = down 1)
    for d in [2, 3, 4]:
        df[f"down_{d}"] = (df["down"] == d).astype(int)

    return df


def train_ep_model(pbp_df: pd.DataFrame) -> dict:
    """
    Train a multinomial EP model on play-by-play data and save it.

    Parameters
    ----------
    pbp_df : DataFrame with columns: game_id, play_number, period, down,
             distance, yards_to_goal, clock_minutes, clock_seconds,
             score_diff, scoring_play, play_type, home, pos_team / offense.

    Returns
    -------
    dict with keys 'model' and 'scaler'.
    """
    df = _label_next_score(pbp_df)
    df = _prepare_ep_features(df)

    X = df[EP_FEATURES].dropna()
    y = df.loc[X.index, "next_score_type"]

    if len(X) < 100:
        logger.warning("Too few plays to train EP model.")
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=2000, C=1.0,
    )
    model.fit(X_scaled, y)

    model_dict = {"model": model, "scaler": scaler}
    joblib.dump(model_dict, EP_MODEL_PATH)
    logger.info(f"EP model trained on {len(X):,} plays → {EP_MODEL_PATH}")
    return model_dict


def load_ep_model() -> dict:
    """Load the saved EP model, or return empty dict if not trained."""
    if EP_MODEL_PATH.exists():
        return joblib.load(EP_MODEL_PATH)
    return {}


def predict_ep(model_dict: dict, play_state: pd.DataFrame) -> np.ndarray:
    """
    Predict expected points for a set of pre-play game states.

    Returns array of shape (n,) with EP values in home-team perspective.
    """
    X = play_state[EP_FEATURES].fillna(0).values
    X_scaled = model_dict["scaler"].transform(X)
    probs = model_dict["model"].predict_proba(X_scaled)
    class_order = model_dict["model"].classes_
    weights = np.array([SCORING_WEIGHTS.get(c, 0) for c in class_order])
    return probs @ weights


def compute_epa(model_dict: dict, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EPA for each play: EP_after - EP_before.
    Returns pbp_df with new columns: ep_before, ep_after, epa.
    """
    df = _prepare_ep_features(pbp_df)
    valid = df[EP_FEATURES].notna().all(axis=1)
    df.loc[valid, "ep_before"] = predict_ep(model_dict, df[valid])
    # ep_after = ep_before of the next play in the same game/half
    df["ep_after"] = (
        df.groupby(["game_id"])["ep_before"].shift(-1)
    )
    # On scoring plays, ep_after = 0 (next possession resets EP to ~0)
    scoring_mask = df["scoring_play"].fillna(False).astype(bool)
    df.loc[scoring_mask, "ep_after"] = 0.0
    df["epa"] = df["ep_after"] - df["ep_before"]
    return df


if __name__ == "__main__":
    import argparse
    from utils.storage import RAW_DIR
    import json

    parser = argparse.ArgumentParser(description="Train EP model from PBP data")
    parser.add_argument("--pbp-file", required=True,
                        help="Path to a JSON file containing PBP records")
    args = parser.parse_args()

    with open(args.pbp_file) as fh:
        pbp_records = json.load(fh)
    pbp = pd.DataFrame(pbp_records)
    train_ep_model(pbp)
