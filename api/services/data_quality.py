"""Data & Model Quality — mirrors ``pages/9_Data_Quality.py``."""
from __future__ import annotations

import pandas as pd

from api.jsonutil import records
from utils.release import load_current_release
from utils.repo_audit import run_repository_audit


def build_data_quality() -> dict:
    """Return the Data & Model Quality payload."""
    report = run_repository_audit()
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
