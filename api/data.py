"""Cached access to the pipeline's Parquet / JSON artifacts.

Streamlit's ``@st.cache_data(ttl=...)`` keeps repeated page loads cheap.  The
API needs the same behaviour, so every artifact is memoised against the file's
modification time: a re-run of the data pipeline invalidates the cache without
a server restart, while a warm cache serves a request in microseconds.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from utils.storage import (
    DATA_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    load_parquet,
)

__all__ = [
    "DATA_DIR",
    "FEATURES_DIR",
    "MODELS_DIR",
    "PROCESSED_DIR",
    "artifact",
    "memo",
    "read_json",
    "parquet",
    "cache_clear",
    "cache_stats",
]

_LOCK = threading.RLock()
_CACHE: dict[str, tuple[object, float | None]] = {}
_HITS = 0
_MISSES = 0


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def artifact(key: str, path: Path, loader: Callable[[], object]) -> object:
    """Memoise ``loader()`` against ``path``'s modification time."""
    global _HITS, _MISSES
    stamp = _mtime(path)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[1] == stamp:
            _HITS += 1
            return cached[0]
    value = loader()
    with _LOCK:
        _MISSES += 1
        _CACHE[key] = (value, stamp)
    return value


def parquet(
    name: str, layer: str = "processed", columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Load a Parquet artifact, memoised until the file changes on disk.

    ``columns`` projects the read. ``feature_matrix.parquet`` carries 266
    columns and the API references 67 of them; pyarrow's decode also costs
    about 2.6x the decoded frame size, so reading the unused 75% cost 117 MB of
    resident memory to produce a 45 MB frame. Projecting drops that to ~30 MB.

    The projected result is cached separately from the full frame, keyed on the
    requested column set.
    """
    folder = PROCESSED_DIR if layer == "processed" else FEATURES_DIR
    path = folder / f"{name}.parquet"
    if columns is None:
        key = f"parquet:{layer}:{name}"
    else:
        key = f"parquet:{layer}:{name}:{','.join(sorted(columns))}"
    return artifact(key, path, lambda: _read_parquet(path, columns))


def _read_parquet(path: Path, columns: Sequence[str] | None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    # Project only against columns the file actually has, so callers can list
    # everything they might use without having to track the artifact's schema.
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
    except Exception:  # noqa: BLE001 - fall back to letting pandas decide
        return pd.read_parquet(path, columns=list(columns))
    wanted = [column for column in columns if column in available]
    if not wanted:
        return pd.DataFrame()
    return pd.read_parquet(path, columns=wanted)


def memo(key: str, loader: Callable[[], object]) -> object:
    """Memoise an expensive external call (e.g. CFBD) for the process lifetime.

    Streamlit's ``@st.cache_data(ttl=3600)`` does the same job on the page side.
    """
    global _HITS, _MISSES
    with _LOCK:
        if key in _CACHE:
            _HITS += 1
            return _CACHE[key][0]
    value = loader()
    with _LOCK:
        _MISSES += 1
        _CACHE[key] = (value, None)
    return value


def read_json(path: Path, default: dict | list | None = None) -> dict | list:
    """Read a JSON artifact, returning ``default`` when it is absent."""

    def _load() -> dict | list:
        if not path.exists():
            return {} if default is None else default
        return json.loads(path.read_text(encoding="utf-8"))

    return artifact(f"json:{path}", path, _load)


def cache_clear() -> None:
    with _LOCK:
        _CACHE.clear()


def cache_stats() -> dict:
    with _LOCK:
        total = _HITS + _MISSES
        return {
            "entries": len(_CACHE),
            "hits": _HITS,
            "misses": _MISSES,
            "hit_rate": (_HITS / total) if total else 0.0,
        }
