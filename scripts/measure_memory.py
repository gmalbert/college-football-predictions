"""Break the API's memory footprint into its stages.

The question this answers: how much of the resident set is the Python
interpreter plus native libraries, and how much is the loaded data? That split
decides whether a 512 MB host is plausible or hopeless, and it is far more
informative than a single number measured on a different OS.
"""
from __future__ import annotations

import ctypes
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def rss_mb() -> float:
    """Resident set size in MB, without needing psutil."""
    if sys.platform == "win32":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        # GetCurrentProcess returns a HANDLE, which is pointer-sized. Without an
        # explicit restype ctypes truncates it to a 32-bit int, the handle
        # becomes invalid, the call fails and the struct stays zeroed.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError(f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
        return counters.WorkingSetSize / 1024 / 1024

    with open("/proc/self/statm") as handle:  # Linux
        return int(handle.read().split()[1]) * 4096 / 1024 / 1024


def stage(label: str, previous: float) -> float:
    gc.collect()
    now = rss_mb()
    print(f"  {label:<46} {now:7.1f} MB   (+{now - previous:.1f})")
    return now


def main() -> int:
    print("=== resident memory by stage ===\n")
    current = stage("bare interpreter", 0.0)

    # The heavy native libraries, in the order the API pulls them in.
    for module in ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "xgboost", "plotly"):
        try:
            __import__(module)
            current = stage(f"+ import {module}", current)
        except ImportError as exc:
            print(f"  + import {module:<34} SKIPPED ({exc})")

    import api.main  # noqa: F401
    current = stage("+ import api.main (all services wired)", current)

    from api.services import (
        data_quality, historical, home, model_performance, preseason,
        team_explorer, total_signals, value_bets, weekly,
    )

    builders = {
        "home": lambda: home.build_home(),
        "weekly": lambda: weekly.build_weekly(timezone_name="UTC"),
        "value_bets": lambda: value_bets.build_value_bets(),
        "team_explorer": lambda: team_explorer.build_team_explorer(),
        "historical": lambda: historical.build_historical(),
        "model_performance": lambda: model_performance.build_model_performance(),
        "preseason": lambda: preseason.build_preseason(),
        "data_quality": lambda: data_quality.build_data_quality(),
        "total_signals": lambda: total_signals.build_total_signals(),
    }
    for name, build in builders.items():
        build()
        current = stage(f"+ build {name}()", current)

    print(f"\n  libraries-only floor (before any data): "
          f"{_libraries_only()} MB est. from the stages above")
    print(f"  final resident set: {current:.0f} MB")
    print("\n  Render's free and $7 tiers are 512 MB; the $25 tier is 2 GB.")
    return 0


def _libraries_only() -> str:
    return "see the '+ import api.main' row"


if __name__ == "__main__":
    raise SystemExit(main())
