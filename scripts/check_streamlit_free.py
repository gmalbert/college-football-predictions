"""Find everything on the API path that depends on Streamlit.

Blocks the ``streamlit`` module with an import hook, then imports every API
service. Whatever fails is what has to be decoupled before the API can ship
without a 57 MB dependency.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class BlockStreamlit:
    """Import hook that refuses ``streamlit`` and anything under it."""

    def find_module(self, name, path=None):  # noqa: D401 - legacy protocol
        if name == "streamlit" or name.startswith("streamlit."):
            return self
        return None

    def find_spec(self, name, path=None, target=None):
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError(f"BLOCKED: {name}")
        return None


sys.meta_path.insert(0, BlockStreamlit())

MODULES = [
    "api.main",
    "api.settings",
    "api.data",
    "api.charts",
    "api.jsonutil",
    "api.services.common",
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

print("=== importing with streamlit blocked ===\n")
failures: dict[str, str] = {}
for name in MODULES:
    try:
        __import__(name)
        print(f"  ok    {name}")
    except ImportError as exc:
        failures[name] = str(exc)
        print(f"  FAIL  {name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures[name] = f"{type(exc).__name__}: {exc}"
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")

if failures:
    print("\n=== traceback for the first failure ===")
    first = next(iter(failures))
    try:
        __import__(first)
    except Exception:  # noqa: BLE001
        traceback.print_exc(limit=8)

print(f"\n{len(MODULES) - len(failures)}/{len(MODULES)} modules import without streamlit")
sys.exit(1 if failures else 0)
