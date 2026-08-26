"""Validate generated artifacts and atomically publish their release metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.release import publish_release  # noqa: E402
from utils.repo_audit import run_repository_audit  # noqa: E402


def main() -> int:
    data = ROOT / "data_files"
    artifacts = [
        data / "processed" / "games.parquet",
        data / "processed" / "lines.parquet",
        data / "features" / "feature_matrix.parquet",
        data / "features" / "model_backtest.parquet",
        data / "models" / "model_metrics.json",
        data / "best_bets_today.json",
        data / "shadow_total_signals.json",
    ]
    audit = run_repository_audit(ROOT)
    metrics = json.loads((data / "models" / "model_metrics.json").read_text(encoding="utf-8"))
    release = publish_release(
        artifacts=artifacts,
        model_version=metrics.get("model_version"),
        audit=audit,
        workflow_run_id=os.getenv("GITHUB_RUN_ID"),
    )
    print(json.dumps(release, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
