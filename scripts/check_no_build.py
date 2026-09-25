"""Simulate a fresh clone: serve the API with frontend/dist absent."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend" / "dist"
STASH = ROOT / "frontend" / "_dist_stashed"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def get(url: str, timeout: int = 20):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}".encode()


def main() -> int:
    if not DIST.exists():
        print("dist already absent")
        return 1
    shutil.move(str(DIST), str(STASH))
    process = subprocess.Popen(
        [str(PYTHON), "-m", "uvicorn", "api.main:app", "--port", "8011", "--log-level", "warning"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(60):
            status, _ = get("http://127.0.0.1:8011/api/health", timeout=3)
            if status:
                break
            time.sleep(1)

        print("=== with frontend/dist absent ===")
        for path in ("/api/health", "/api/home", "/", "/Weekly_Predictions"):
            status, body = get(f"http://127.0.0.1:8011{path}", timeout=60)
            preview = body[:110].decode("utf-8", "replace").replace("\n", " ")
            print(f"  {path:<22} {status}  {preview}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        shutil.move(str(STASH), str(DIST))
        print("\ndist restored:", DIST.exists())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
