"""Lightweight artifact manifests, fingerprints, drift, and promotion gates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArtifactManifest:
    model_name: str
    version: str
    created_at: str
    git_sha: str | None
    training_start: str | None
    training_end: str | None
    feature_names: tuple[str, ...]
    feature_schema_hash: str
    data_fingerprint: str
    parameters: Mapping[str, object]
    metrics: Mapping[str, object]
    prediction_contract: str
    status: str = "challenger"

    def to_dict(self) -> dict:
        return asdict(self)


def dataframe_fingerprint(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] | None = None,
) -> str:
    """Stable content fingerprint independent of the input row order."""
    selected = frame.copy()
    if key_columns:
        keys = [column for column in key_columns if column in selected.columns]
        if keys:
            selected = selected.sort_values(keys)
    selected = selected.reindex(sorted(selected.columns), axis=1).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(selected, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def schema_fingerprint(frame: pd.DataFrame, features: Sequence[str]) -> str:
    schema = [
        (feature, str(frame[feature].dtype) if feature in frame.columns else "missing")
        for feature in features
    ]
    return hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()


def create_manifest(
    *,
    model_name: str,
    version: str,
    training_frame: pd.DataFrame,
    features: Sequence[str],
    parameters: Mapping[str, object],
    metrics: Mapping[str, object],
    prediction_contract: str,
    git_sha: str | None = None,
    time_column: str = "start_date",
) -> ArtifactManifest:
    timestamps = (
        pd.to_datetime(training_frame[time_column], errors="coerce", utc=True)
        if time_column in training_frame.columns else pd.Series(dtype="datetime64[ns, UTC]")
    )
    valid_times = timestamps.dropna()
    return ArtifactManifest(
        model_name=model_name,
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha,
        training_start=valid_times.min().isoformat() if len(valid_times) else None,
        training_end=valid_times.max().isoformat() if len(valid_times) else None,
        feature_names=tuple(features),
        feature_schema_hash=schema_fingerprint(training_frame, features),
        data_fingerprint=dataframe_fingerprint(
            training_frame[[c for c in ["game_id", *features] if c in training_frame.columns]],
            key_columns=["game_id"],
        ),
        parameters=dict(parameters),
        metrics=dict(metrics),
        prediction_contract=prediction_contract,
    )


def save_manifest(manifest: ArtifactManifest, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")
    temporary.replace(destination)
    return destination


def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI using quantile bins learned from the reference sample."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < bins or not len(cur):
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            rows.append(
                {"feature": feature, "psi": np.nan, "status": "missing", "current_null_pct": 1.0}
            )
            continue
        psi = population_stability_index(reference[feature], current[feature])
        null_pct = float(current[feature].isna().mean())
        status = "alert" if (np.isfinite(psi) and psi >= 0.25) or null_pct >= 0.50 else (
            "watch" if (np.isfinite(psi) and psi >= 0.10) or null_pct >= 0.20 else "ok"
        )
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "status": status,
                "reference_null_pct": float(reference[feature].isna().mean()),
                "current_null_pct": null_pct,
                "reference_mean": float(pd.to_numeric(reference[feature], errors="coerce").mean()),
                "current_mean": float(pd.to_numeric(current[feature], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "psi"], ascending=[True, False])


def promotion_decision(
    gates: Sequence[Mapping[str, object]],
    *,
    require_all: bool = True,
) -> dict[str, object]:
    passed = [bool(gate.get("passed")) for gate in gates]
    promote = all(passed) if require_all else bool(passed) and sum(passed) > len(passed) / 2
    return {
        "decision": "promote" if promote else "hold",
        "passed": int(sum(passed)),
        "total": len(passed),
        "failed_gates": [gate.get("name") for gate in gates if not gate.get("passed")],
    }
