"""Run the whole parity verification pipeline.

Starts nothing itself — it assumes the two servers are already up:

* Streamlit   on ``http://127.0.0.1:8501``
* FastAPI     on ``http://127.0.0.1:8000`` (also serving ``frontend/dist``)

Steps: byte-compile, refresh the Streamlit snapshots, compare content and live
chrome, run the pytest suites, then print the headline benchmark numbers.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

STEPS: list[tuple[str, list[str]]] = [
    (
        "Byte-compile every Python module",
        ["-m", "pytest", "tests/test_api_parity.py", "-q", "-k", "byte_compiles"],
    ),
    ("Generate Streamlit snapshots", ["scripts/parity_snapshot.py"]),
    (
        # The Data Quality page serves the audit the workflow publishes, so it
        # has to be regenerated here the same way the pipeline regenerates it —
        # otherwise the page is compared against a report older than the
        # artifacts it describes.
        "Regenerate the published audit report",
        ["scripts/audit_pipeline.py", "--output", "data_files/audit_report.json"],
    ),
    ("Parity check (content + live chrome)", ["scripts/parity_check.py", "--content", "--live", "--json"]),
    ("API contract tests", ["-m", "pytest", "tests/test_api_parity.py", "-q"]),
    ("Audit-projection equivalence tests", ["-m", "pytest", "tests/test_audit_projection.py", "-q"]),
    ("Playwright end-to-end tests", ["-m", "pytest", "tests/test_web_e2e.py", "-q"]),
]


def main() -> int:
    started = time.perf_counter()
    failures = 0
    for label, args in STEPS:
        print(f"\n{'=' * 72}\n>>> {label}\n{'=' * 72}")
        result = subprocess.run([PYTHON, *args], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            failures += 1
            print(f"!!! {label} failed (exit {result.returncode})")

    elapsed = time.perf_counter() - started
    print(f"\n{'=' * 72}")
    print(f"{len(STEPS) - failures}/{len(STEPS)} steps passed in {elapsed:.0f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
