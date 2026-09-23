"""Per-artifact parquet read cost vs. resulting frame size.

Read cost being ~2.6x the decoded frame is what dominates the API's resident
set, so this ranks the artifacts by what they actually cost to load.
"""
from __future__ import annotations

import ctypes
import gc
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def rss_mb() -> float:
    kernel32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    c = PMC()
    c.cb = ctypes.sizeof(c)
    psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1024 / 1024


ARTIFACTS = [
    ("feature_matrix", "features"),
    ("model_backtest", "features"),
    ("games", "processed"),
    ("line_snapshots", "processed"),
    ("team_game_stats", "processed"),
    ("advanced_stats", "processed"),
    ("ratings", "processed"),
    ("elo_ratings", "processed"),
    ("returning_production", "processed"),
    ("transfer_portal", "processed"),
    ("feature_observations", "processed"),
]

print(f"{'artifact':<22}{'rows':>8}{'cols':>6}{'file MB':>9}{'frame MB':>10}{'read MB':>9}{'ratio':>7}{'used cols':>10}")
print("-" * 82)

kept = []
for name, layer in ARTIFACTS:
    path = ROOT / "data_files" / layer / f"{name}.parquet"
    if not path.exists():
        print(f"{name:<22}{'MISSING':>8}")
        continue
    gc.collect()
    before = rss_mb()
    frame = pd.read_parquet(path)
    gc.collect()
    cost = rss_mb() - before
    fmem = frame.memory_usage(deep=True).sum() / 1024 / 1024
    fmem = frame.memory_usage(deep=True).sum() / 1024 / 1024
    kept.append(frame)
    print(
        f"{name:<22}{len(frame):>8,}{frame.shape[1]:>6}{path.stat().st_size / 1e6:>9.1f}"
        f"{fmem:>10.1f}{cost:>9.1f}{(cost / fmem if fmem else 0):>7.2f}x"
    )

print(f"\n  total resident after loading all of them: {rss_mb():.0f} MB")
