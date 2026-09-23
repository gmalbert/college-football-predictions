"""How much of feature_matrix.parquet does the API actually read?"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.feature_engine import (  # noqa: E402
    SPREAD_FEATURES, TOTAL_COVER_FEATURES, TOTAL_FEATURES, WIN_FEATURES,
)

MODEL_FEATURES = list(dict.fromkeys(
    WIN_FEATURES + SPREAD_FEATURES + TOTAL_FEATURES + TOTAL_COVER_FEATURES
))

# Columns the pages display or filter on, plus keys and the columns
# predict_for_display reads to decide completed vs upcoming.
DISPLAY_AND_KEYS = [
    "game_id", "season", "week", "start_date", "neutral_site",
    "home_team", "away_team", "home_conference", "away_conference",
    "conference_game", "home_score", "away_score",
    "home_margin", "total_points", "home_win",
    "market_spread", "market_total", "home_moneyline", "away_moneyline",
    "market_spread_open", "market_spread_move",
    "market_total_open", "market_total_move",
    "prediction_scope", "home_points_for_l5", "away_points_for_l5",
]

path = ROOT / "data_files" / "features" / "feature_matrix.parquet"
frame = pd.read_parquet(path)
all_columns = list(frame.columns)

needed = list(dict.fromkeys(MODEL_FEATURES + DISPLAY_AND_KEYS))
present = [c for c in needed if c in all_columns]
unused = [c for c in all_columns if c not in set(present)]

print(f"  columns in the artifact      : {len(all_columns)}")
print(f"  columns the API needs        : {len(present)}")
print(f"  columns never referenced     : {len(unused)}  ({len(unused) / len(all_columns):.0%})")
print(f"  frame memory, full           : {frame.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

subset = frame[present]
print(f"  frame memory, projected      : {subset.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

print(f"\n  projection would cut the resident frame by "
      f"{100 * (1 - subset.memory_usage(deep=True).sum() / frame.memory_usage(deep=True).sum()):.0f}%")

print(f"\n  a sample of the {len(unused)} unused columns:")
for name in unused[:25]:
    print(f"    {name}")
if len(unused) > 25:
    print(f"    ... and {len(unused) - 25} more")
