"""Prove the API runs in an environment that has no Streamlit at all.

Builds a throwaway virtual environment, installs only requirements-api.txt,
then asserts that streamlit is genuinely absent and that every API service
still works. This is the check that gates a serverless/container deployment
where a 57 MB (plus transitive) dependency would be unacceptable.

Takes a couple of minutes the first time; the venv is cached and reused.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = Path(tempfile.gettempdir()) / "parity-api-only-venv"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python"
    )


def run(args, cwd=ROOT, label=None, expect_failure=False):
    result = subprocess.run([str(a) for a in args], cwd=str(cwd), capture_output=True, text=True)
    if label:
        succeeded = result.returncode != 0 if expect_failure else result.returncode == 0
        print(f"  {'PASS' if succeeded else 'FAIL'}  {label}")
        if not succeeded:
            for line in (result.stdout + result.stderr).strip().splitlines()[-10:]:
                print(f"        {line}")
    return result


def main() -> int:
    python = venv_python(VENV)

    if not python.exists():
        print(f"creating clean venv at {VENV}")
        if VENV.exists():
            shutil.rmtree(VENV, ignore_errors=True)
        run([sys.executable, "-m", "venv", str(VENV)])
        print("installing requirements-api.txt (no streamlit)\n")
        install = run([python, "-m", "pip", "install", "--quiet",
                       "-r", str(ROOT / "requirements-api.txt")])
        if install.returncode != 0:
            for line in (install.stdout + install.stderr).strip().splitlines()[-15:]:
                print("   ", line)
            return 1
    else:
        print(f"reusing venv at {VENV}\n")

    checks: list[bool] = []

    # 1. Neither the ML stack nor streamlit may be installed.
    for heavy in ("streamlit", "sklearn", "xgboost", "scipy"):
        probe = run([python, "-c", f"import {heavy}"],
                    label=f"{heavy} is absent",
                    expect_failure=True)
        checks.append(probe.returncode != 0)

    # 2. The API imports and serves without them.
    checks.append(
        run([python, "-c",
             "import api.main; print(len(api.main.app.routes), 'routes')"],
            label="api.main imports with no ML stack").returncode == 0
    )

    # 3. Every page service actually produces a payload.
    payload_check = """
import sys
sys.path.insert(0, '.')
from api.services import (home, weekly, value_bets, team_explorer, historical,
                          model_performance, preseason, data_quality, total_signals,
                          win_probability)
built = {
    'home': home.build_home(),
    'weekly': weekly.build_weekly(timezone_name='UTC'),
    'value_bets': value_bets.build_value_bets(),
    'team_explorer': team_explorer.build_team_explorer(),
    'historical': historical.build_historical(),
    'model_performance': model_performance.build_model_performance(),
    'preseason': preseason.build_preseason(),
    'data_quality': data_quality.build_data_quality(),
    'total_signals': total_signals.build_total_signals(),
}
for name, payload in built.items():
    assert payload.get('title'), name
print('built', len(built), 'page payloads without streamlit')
"""
    checks.append(run([python, "-c", payload_check],
                      label="all 9 offline pages build payloads").returncode == 0)

    # 4. The CFBD-backed page also imports (needs the network, not asserted).
    checks.append(
        run([python, "-c", "import api.services.win_probability; print('ok')"],
            label="win_probability (cfbd path) imports").returncode == 0
    )

    # 5. Size of the API-only environment.
    if python.exists():
        size = sum(
            f.stat().st_size
            for f in (VENV / ("Lib" if sys.platform == "win32" else "lib")).rglob("*")
            if f.is_file()
        )
        print(f"\n  API-only environment size: {size / 1024 / 1024:.0f} MB")

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
