"""Release metadata and readiness contracts for generated application artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from utils.storage import DATA_DIR, RELEASES_DIR, atomic_write_json


RELEASE_PATH = RELEASES_DIR / "current_release.json"


def market_release_statuses(metrics: dict) -> dict[str, dict]:
    """Return fail-closed readiness per separately evaluated market strategy."""
    gate_results = {
        item.get("name"): bool(item.get("passed"))
        for item in metrics.get("release_gates", [])
    }
    clv = metrics.get("prospective_clv", {})
    requirements = {
        "moneyline": (("out_of_sample_games", "brier", "calibration", "win_market_baseline"), "moneyline:all"),
        "spread": (("out_of_sample_games", "spread_market_baseline"), "spread:all"),
        "total": (("out_of_sample_games", "total_market_baseline"), "total:over"),
    }
    results = {}
    for market, (market_gates, clv_key) in requirements.items():
        baseline_passed = all(gate_results.get(gate, False) for gate in market_gates)
        clv_passed = bool(clv.get(clv_key, {}).get("passed", False))
        if market == "total":
            total_strategy = metrics.get("total_cover_model", {}).get("strategy_release", {})
            baseline_passed = (
                baseline_passed and bool(total_strategy.get("retrospective_gates_passed"))
                and "over" in total_strategy.get("eligible_sides", [])
            )
        if baseline_passed and clv_passed:
            status = "promote"
        elif market == "total" and baseline_passed:
            status = "shadow"
        else:
            status = "hold"
        results[market] = {
            "status": status,
            "required_gates": list(market_gates),
            "baseline_passed": baseline_passed,
            "clv_key": clv_key,
            "clv_passed": clv_passed,
        }
    return results


def file_fingerprint(paths: Iterable[Path]) -> str:
    """Create a stable fingerprint from the exact published files."""
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        digest.update(str(path.relative_to(DATA_DIR)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def publish_release(
    *,
    artifacts: Iterable[Path],
    model_version: str | None,
    audit: dict,
    workflow_run_id: str | None = None,
) -> dict:
    """Publish a release manifest only after all artifacts passed validation."""
    published = [Path(path) for path in artifacts]
    missing = [str(path) for path in published if not path.exists()]
    if missing:
        raise FileNotFoundError(f"release artifacts missing: {missing}")
    if audit.get("summary", {}).get("fail", 0):
        raise ValueError("cannot publish a release with failed audit checks")
    payload = {
        "release_id": uuid4().hex,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "workflow_run_id": workflow_run_id,
        "artifact_fingerprint": file_fingerprint(published),
        "artifacts": [str(path.relative_to(DATA_DIR)) for path in published],
        "audit_summary": audit.get("summary", {}),
        "status": "shadow" if audit.get("release_status") == "shadow" else "hold",
    }
    atomic_write_json(RELEASE_PATH, payload)
    return payload


def load_current_release() -> dict:
    if not RELEASE_PATH.exists():
        return {}
    return json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
