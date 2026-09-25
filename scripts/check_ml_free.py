"""Which heavy libraries does the API still need after precomputing predictions?

Blocks sklearn / xgboost / scipy / streamlit with an import hook and imports
every API module. Whatever still fails is what genuinely has to stay in the
API's dependency set.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BLOCKED = ("sklearn", "xgboost", "scipy", "streamlit")


class Blocker:
    def __init__(self, names):
        self.names = tuple(names)

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in self.names:
            raise ImportError(f"BLOCKED: {name}")
        return None


sys.meta_path.insert(0, Blocker(BLOCKED))

MODULES = [
    "api.main",
    "api.settings",
    "api.data",
    "api.columns",
    "api.services.home",
    "api.services.weekly",
    "api.services.value_bets",
    "api.services.team_explorer",
    "api.services.historical",
    "api.services.model_performance",
    "api.services.win_probability",
    "api.services.preseason",
    "api.services.data_quality",
    "api.services.total_signals",
]

print(f"blocked: {', '.join(BLOCKED)}\n")
failures = {}
for name in MODULES:
    try:
        __import__(name)
        print(f"  ok    {name}")
    except ImportError as exc:
        failures[name] = str(exc)
        print(f"  FAIL  {name}: {exc}")

if failures:
    print("\nfirst traceback:")
    import traceback

    try:
        __import__(next(iter(failures)))
    except Exception:  # noqa: BLE001
        traceback.print_exc(limit=12)

print(f"\n{len(MODULES) - len(failures)}/{len(MODULES)} modules import without {BLOCKED}")
sys.exit(1 if failures else 0)
