"""CLI for the repository data/model audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.repo_audit import run_repository_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit data/model release contracts")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on failed checks")
    args = parser.parse_args()
    report = run_repository_audit(ROOT)
    rendered = json.dumps(report, indent=2, default=str)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.strict and report.get("strict_failures", report["summary"]["fail"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
