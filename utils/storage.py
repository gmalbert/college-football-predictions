"""utils/storage.py — Read/write helpers for the data pipeline."""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data_files"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"

for _d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def save_raw_json(data: list | dict, name: str) -> Path:
    path = RAW_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, default=str)
    return path


def save_parquet(df: pd.DataFrame, name: str, layer: str = "processed") -> Path:
    folder = PROCESSED_DIR if layer == "processed" else FEATURES_DIR
    path = folder / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_parquet(name: str, layer: str = "processed") -> pd.DataFrame:
    folder = PROCESSED_DIR if layer == "processed" else FEATURES_DIR
    path = folder / f"{name}.parquet"
    return pd.read_parquet(path)
