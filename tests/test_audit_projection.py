"""The audit must reach identical conclusions on projected frames.

``utils/contracts.py`` skips checks whose columns are absent, so handing
``run_repository_audit`` a projected frame can silently turn a warning into a
"pass". That happened once: api/columns.FEATURE_MATRIX_COLUMNS omits the
leakage-risk columns, and the Data Quality page started reporting "No known
unsafe season aggregates present" instead of the warning it should.

This test compares the audit run over the projected frames the API actually
uses against the audit run over the full artifacts, and fails on any
difference. It is the guard that catches a projection which is too narrow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.services.data_quality import AUDITED_ARTIFACTS  # noqa: E402
from api.data import parquet  # noqa: E402
from utils.repo_audit import run_repository_audit  # noqa: E402

DATA_AVAILABLE = (PROJECT_ROOT / "data_files" / "features" / "feature_matrix.parquet").exists()
pytestmark = pytest.mark.skipif(not DATA_AVAILABLE, reason="pipeline artifacts not present")


def _normalise(report: dict) -> list[tuple]:
    """Comparable view of an audit result, ignoring the generation timestamp."""
    return sorted(
        (check["name"], check["status"], check["message"])
        for check in report["checks"]
    )


def _projected_frames() -> dict:
    frames = {}
    for key, (name, layer, columns) in AUDITED_ARTIFACTS.items():
        try:
            frames[key] = parquet(name, layer=layer, columns=columns)
        except FileNotFoundError:
            continue
    return frames


def test_projected_audit_matches_full_audit() -> None:
    full = run_repository_audit()
    projected = run_repository_audit(frames=_projected_frames())

    assert _normalise(projected) == _normalise(full), (
        "the audit reached different conclusions on projected frames — a "
        "projection is dropping a column the audit reasons about"
    )


def test_projected_audit_still_reports_leakage_columns() -> None:
    """The specific regression: this check must not become a false all-clear."""
    report = run_repository_audit(frames=_projected_frames())
    check = next(c for c in report["checks"] if c["name"] == "unsafe_exploration_columns")
    assert check["status"] == "warn", check
    assert "sp_plus_diff" in check["message"], check


def test_projected_audit_summary_matches() -> None:
    full = run_repository_audit()
    projected = run_repository_audit(frames=_projected_frames())
    assert projected["summary"] == full["summary"]
