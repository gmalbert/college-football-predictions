"""Data & Model Quality — mirrors ``pages/9_Data_Quality.py``."""
from __future__ import annotations

import pandas as pd

from api.columns import (
    AUDIT_FEATURE_MATRIX_COLUMNS,
    GAMES_COLUMNS,
    LINE_SNAPSHOT_COLUMNS,
)
from api.data import DATA_DIR, parquet, read_json
from api.jsonutil import records
from utils.release import load_current_release
from utils.repo_audit import merge_live_checks, run_repository_audit

# The weekly workflow runs the same audit and commits its output here
# (scripts/audit_pipeline.py --output). Serving that report means the page does
# no artifact reads at all in the normal case — the checks are already computed
# and this is just a small JSON read.
AUDIT_REPORT_PATH = DATA_DIR / "audit_report.json"

# Only used when no report has been published yet (e.g. before the first
# pipeline run after a checkout).
AUDITED_ARTIFACTS: dict[str, tuple[str, str, list[str] | None]] = {
    "games": ("games", "processed", GAMES_COLUMNS),
    "feature_matrix": ("feature_matrix", "features", AUDIT_FEATURE_MATRIX_COLUMNS),
    "team_game_stats": ("team_game_stats", "processed", None),
    "line_snapshots": ("line_snapshots", "processed", LINE_SNAPSHOT_COLUMNS),
    "feature_observations": ("feature_observations", "processed", None),
    "model_backtest": ("model_backtest", "features", None),
}


def _cached_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for key, (name, layer, columns) in AUDITED_ARTIFACTS.items():
        try:
            frames[key] = parquet(name, layer=layer, columns=columns)
        except FileNotFoundError:
            continue
    return frames


def _audit_report() -> dict:
    """The pipeline's audit, with only the live-state checks recomputed.

    Everything else is a property of the committed artifacts, so it is read
    from the report the workflow published rather than recomputed — which is
    what keeps this page from loading six Parquet files per request.
    """
    published = read_json(AUDIT_REPORT_PATH, default={})
    if isinstance(published, dict) and published.get("checks"):
        return merge_live_checks(published, DATA_DIR)
    return run_repository_audit(frames=_cached_frames())


def build_data_quality() -> dict:
    """Return the Data & Model Quality payload."""
    report = _audit_report()
    release = load_current_release()
    summary = report["summary"]

    captions: list[str] = []
    warnings: list[str] = []
    if release:
        captions.append(
            f"Release {release.get('release_id', 'unknown')[:12]} · "
            f"{release.get('status', 'hold').upper()} · "
            f"generated {release.get('generated_at', 'unknown')}"
        )
    else:
        warnings.append(
            "No release metadata is published. Treat generated artifacts as unreleased."
        )

    checks = pd.DataFrame(report["checks"])
    table = None
    if not checks.empty:
        checks["Status"] = checks["status"].map(
            {"pass": "✅ Pass", "warn": "⚠️ Warning", "fail": "❌ Fail"}
        )
        checks["value"] = checks["value"].map(
            lambda value: "—" if value is None else str(value)
        )
        display = checks[["Status", "name", "message", "value"]].rename(
            columns={"name": "Check", "message": "Details", "value": "Value"}
        )
        table = {
            "columns": [str(column) for column in display.columns],
            "rows": records(display),
        }

    return {
        "page": "data_quality",
        "title": "🛡️ Data & Model Quality",
        "caption": (
            "Release-contract checks for grain, point-in-time safety, "
            "freshness, and evaluation scope."
        ),
        "metrics": [
            {"label": "Passing", "value": str(summary["pass"]), "delta": None, "help": None},
            {"label": "Warnings", "value": str(summary["warn"]), "delta": None, "help": None},
            {"label": "Failures", "value": str(summary["fail"]), "delta": None, "help": None},
        ],
        "captions": captions,
        "warnings": warnings,
        "table": table,
        "info": (
            "A warning does not automatically block the dashboard. A failed grain, "
            "point-in-time, or out-of-sample evaluation check should block model promotion."
        ),
        "footer": True,
    }
