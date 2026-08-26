"""Import a timestamped contextual provider export into the canonical feature store.

The input must be CSV or Parquet and include: entity_id, entity_type,
feature_name, value, available_at, and source_version. This intentionally
rejects undated injury/weather/ranking exports instead of guessing their
historical availability.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.feature_observations import append_feature_observations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Import point-in-time contextual observations")
    parser.add_argument("input", type=Path, help="CSV or Parquet provider export")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    frame = pd.read_parquet(args.input) if args.input.suffix.lower() == ".parquet" else pd.read_csv(args.input)
    destination = append_feature_observations(frame)
    print(f"Imported {len(frame):,} observations -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
