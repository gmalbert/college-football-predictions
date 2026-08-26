"""utils/storage.py — Read/write helpers for the data pipeline."""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

DATA_DIR = Path(__file__).resolve().parent.parent / "data_files"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
RELEASES_DIR = DATA_DIR / "releases"

for _d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, RELEASES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def save_raw_json(data: list | dict, name: str) -> Path:
    path = RAW_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, default=str)
    return path


def save_immutable_raw_json(
    data: list | dict,
    *,
    source: str,
    season: int,
    captured_at: datetime | None = None,
    run_id: str | None = None,
) -> tuple[Path, str, datetime]:
    """Persist a source response in an immutable capture-time partition."""
    captured = captured_at or datetime.now(timezone.utc)
    captured = captured.astimezone(timezone.utc)
    identifier = run_id or uuid4().hex
    directory = RAW_DIR / "snapshots" / source / f"season={season}" / (
        f"captured_date={captured:%Y-%m-%d}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{captured:%Y%m%dT%H%M%S%fZ}_{identifier}.json"
    path.write_text(json.dumps(data, default=str), encoding="utf-8")
    return path, identifier, captured


def atomic_write_json(path: Path, payload: dict | list) -> Path:
    """Write JSON atomically so readers never see a partial release file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)
    return path


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Write a parquet artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path


def save_parquet(df: pd.DataFrame, name: str, layer: str = "processed") -> Path:
    folder = PROCESSED_DIR if layer == "processed" else FEATURES_DIR
    path = folder / f"{name}.parquet"
    return atomic_write_parquet(df, path)


def load_parquet(name: str, layer: str = "processed") -> pd.DataFrame:
    folder = PROCESSED_DIR if layer == "processed" else FEATURES_DIR
    path = folder / f"{name}.parquet"
    return pd.read_parquet(path)
