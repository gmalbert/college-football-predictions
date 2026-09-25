"""Smoke-test a fresh checkout of the committed tree.

Adds a detached worktree at HEAD, runs the API contract suite against it, and
removes the worktree again. Confirms nothing importable was left uncommitted.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
WORKTREE = Path(tempfile.gettempdir()) / "parity-fresh-checkout"


def run(args, cwd, label):
    result = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
    ok = result.returncode == 0
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        tail = (result.stdout + result.stderr).strip().splitlines()[-12:]
        for line in tail:
            print(f"        {line}")
    return ok


def main() -> int:
    if WORKTREE.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(WORKTREE)],
                       cwd=str(ROOT), capture_output=True)

    added = subprocess.run(["git", "worktree", "add", "--detach", str(WORKTREE), "HEAD"],
                           cwd=str(ROOT), capture_output=True, text=True)
    if added.returncode != 0:
        print("could not create worktree:", added.stderr.strip())
        return 1
    print(f"fresh checkout at {WORKTREE}\n")

    checks = []
    try:
        checks.append(run([str(PYTHON), "-c", "import api.main; print(len(api.main.app.routes))"],
                          WORKTREE, "api.main imports from the committed tree"))
        checks.append(run([str(PYTHON), "-c",
                           "from api.settings import load_settings; "
                           "print(load_settings({'TAILGATE_WORKERS': '3'}).workers)"],
                          WORKTREE, "api.settings imports and parses"))
        checks.append(run([str(PYTHON), "-c",
                           "from utils.cfbd_client import parse_win_probability_rows; "
                           "from utils.ui_components import browser_timezone; print('ok')"],
                          WORKTREE, "the two prerequisite utils helpers are present"))
        checks.append(run([str(PYTHON), "-m", "pytest", "tests/test_api_parity.py", "-q",
                           "--ignore=tests/test_web_e2e.py"],
                          WORKTREE, "API contract suite passes in the fresh checkout"))
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(WORKTREE)],
                       cwd=str(ROOT), capture_output=True)
        if WORKTREE.exists():
            shutil.rmtree(WORKTREE, ignore_errors=True)
        print("\nworktree removed")

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
