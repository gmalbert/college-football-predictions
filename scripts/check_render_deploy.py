"""Rehearse the Render deployment locally.

Runs what the blueprint runs, in the deployment environment rather than the
development one:

1. install requirements-api.txt into a throwaway venv (no streamlit, no
   scikit-learn, no XGBoost, no scipy — see scripts/check_api_isolated.py)
2. build the frontend with ``npm ci && npm run build`` from the repo root,
   the way Render's build command does
3. start ``python -m api`` bound to 0.0.0.0 and an injected ``PORT``
4. verify the API answers, the SPA is served at the same origin, and the
   artifact-backed pages render

This is the closest thing to a deploy that can be done without a Render
account, so a failure here is a failure there.

Usage::

    python scripts/check_render_deploy.py
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = Path(tempfile.gettempdir()) / "parity-render-venv"
DIST = ROOT / "frontend" / "dist"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def step(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail and not ok:
        for line in detail.strip().splitlines()[-8:]:
            print(f"        {line}")
    return ok


def resolve(executable: str) -> str:
    """Resolve npm/node, which are .cmd shims on Windows."""
    found = shutil.which(executable)
    if found:
        return found
    for suffix in (".cmd", ".exe", ".bat"):
        found = shutil.which(executable + suffix)
        if found:
            return found
    return executable


def run(args, cwd=ROOT, timeout=1800):
    args = list(args)
    args[0] = resolve(str(args[0]))
    return subprocess.run(
        [str(a) for a in args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )


def get(url, timeout=60):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}".encode()


def main() -> int:
    checks: list[bool] = []
    python = venv_python(VENV)

    # ── 1. deployment environment ───────────────────────────────────────────
    print("=== 1. build the deployment environment (requirements-api.txt) ===")
    if not python.exists():
        if VENV.exists():
            shutil.rmtree(VENV, ignore_errors=True)
        run([sys.executable, "-m", "venv", str(VENV)])
        install = run([python, "-m", "pip", "install", "--quiet",
                       "-r", str(ROOT / "requirements-api.txt")])
        checks.append(step("pip install -r requirements-api.txt", install.returncode == 0,
                           install.stdout + install.stderr))
    else:
        print(f"  reusing {VENV}")

    for heavy in ("streamlit", "sklearn", "xgboost", "scipy"):
        probe = run([python, "-c", f"import {heavy}"])
        checks.append(step(f"{heavy} is not installed", probe.returncode != 0))

    # ── 2. frontend build, from the repo root as Render does ────────────────
    print("\n=== 2. npm ci && npm run build (Render's form) ===")
    ci = run(["npm", "ci", "--prefix", "frontend", "--no-audit", "--no-fund"])
    checks.append(step("npm ci --prefix frontend", ci.returncode == 0, ci.stdout + ci.stderr))

    build = run(["npm", "run", "build", "--prefix", "frontend"])
    checks.append(step("npm run build --prefix frontend", build.returncode == 0,
                       build.stdout + build.stderr))
    checks.append(step("frontend/dist/index.html exists", (DIST / "index.html").exists()))

    # ── 3. start exactly as the blueprint does ─────────────────────────────
    print("\n=== 3. python -m api with an injected PORT ===")
    port = free_port()
    env = {
        **os.environ,
        "PORT": str(port),
        "TAILGATE_HOST": "0.0.0.0",
        "TAILGATE_WARM_CACHE": "true",
        "TAILGATE_LOG_LEVEL": "info",
        "PYTHONIOENCODING": "utf-8",
    }
    process = subprocess.Popen(
        [str(python), "-m", "api"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        ready = False
        deadline = time.time() + 240
        while time.time() < deadline:
            status, _ = get(f"{base}/api/health", timeout=5)
            if status:
                ready = True
                break
            if process.poll() is not None:
                break
            time.sleep(2)
        checks.append(step(f"server bound PORT={port} and answered /api/health", ready))

        if ready:
            print("\n=== 4. smoke test the deployed shape ===")
            status, body = get(f"{base}/api/health")
            health = json.loads(body)
            checks.append(step("/api/health returns ok", status == 200 and health["status"] == "ok",
                               str(health)[:200]))

            status, body = get(f"{base}/")
            html = body.decode("utf-8", "replace")
            checks.append(step("/ serves the built SPA", status == 200 and "<div id=\"root\">" in html,
                               html[:200]))

            status, _ = get(f"{base}/assets/")  # directory listing is not a thing
            status, body = get(f"{base}/Weekly_Predictions")
            checks.append(step("deep link serves the SPA shell", status == 200))

            pages = [
                "/api/home", "/api/weekly-predictions?tz=UTC", "/api/value-bets",
                "/api/team-explorer", "/api/historical-analysis",
                "/api/model-performance", "/api/preseason-outlook",
                "/api/data-quality", "/api/total-market-signals",
            ]
            bad = []
            for path in pages:
                status, body = get(f"{base}{path}", timeout=120)
                if status != 200:
                    bad.append(f"{path} -> {status}")
            checks.append(step(f"all {len(pages)} page endpoints return 200", not bad, "\n".join(bad)))

            # The asset the SPA actually needs must be reachable.
            import re

            match = re.search(r'src="(/assets/[^"]+)"', html)
            if match:
                status, _ = get(f"{base}{match.group(1)}")
                checks.append(step("the hashed JS bundle is served", status == 200))
            else:
                checks.append(step("the hashed JS bundle is referenced", False, html[:200]))
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
