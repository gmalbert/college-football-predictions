"""Timestamped contextual feature store and point-in-time game materialization."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.contracts import ensure_utc, validate_feature_observations
from utils.storage import PROCESSED_DIR, atomic_write_parquet


OBSERVATIONS_PATH = PROCESSED_DIR / "feature_observations.parquet"


def append_feature_observations(observations: pd.DataFrame, path: Path = OBSERVATIONS_PATH) -> Path:
    """Append typed provider observations without overwriting prior revisions."""
    frame = observations.copy()
    frame["available_at"] = ensure_utc(frame["available_at"])
    validate_feature_observations(frame).raise_for_errors()
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=frame.columns)
    combined = pd.concat([existing, frame], ignore_index=True)
    keys = ["entity_id", "entity_type", "feature_name", "available_at", "source_version"]
    combined = combined.drop_duplicates(keys, keep="last").sort_values("available_at")
    validate_feature_observations(combined).raise_for_errors()
    return atomic_write_parquet(combined, path)


def attach_game_observations(games: pd.DataFrame, observations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Backward-as-of join game-level observations using the game prediction time."""
    result = games.copy()
    if observations is None:
        observations = pd.read_parquet(OBSERVATIONS_PATH) if OBSERVATIONS_PATH.exists() else pd.DataFrame()
    if observations.empty or "start_date" not in result.columns:
        result["context_observation_coverage"] = 0.0
        return result
    source = observations[observations["entity_type"].eq("game")].copy()
    if source.empty:
        result["context_observation_coverage"] = 0.0
        return result
    source["available_at"] = ensure_utc(source["available_at"])
    result["prediction_time"] = ensure_utc(result["start_date"])
    added = []
    for feature, group in source.groupby("feature_name", sort=True):
        right = group[["entity_id", "available_at", "value"]].copy()
        values: dict[str, float] = {}
        for game_id, prediction_time in zip(result["game_id"], result["prediction_time"]):
            eligible = right[
                right["entity_id"].astype(str).eq(str(game_id))
                & (right["available_at"] <= prediction_time)
            ]
            if not eligible.empty:
                values[str(game_id)] = eligible.sort_values("available_at").iloc[-1]["value"]
        result[feature] = pd.to_numeric(result["game_id"].astype(str).map(values), errors="coerce")
        added.append(feature)
    result["context_observation_coverage"] = (
        result[added].notna().mean(axis=1) if added else 0.0
    )
    return result
