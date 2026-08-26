"""Capture a forward-only CFBD odds snapshot for movement and CLV tracking."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.cfbd_client import get_lines  # noqa: E402
from utils.odds_ingestion import append_line_snapshots, normalize_cfbd_line_snapshots  # noqa: E402
from utils.seasons import current_cfb_season  # noqa: E402
from utils.storage import PROCESSED_DIR, save_raw_json, save_immutable_raw_json  # noqa: E402


def _plain(value):
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Append the latest CFBD sportsbook snapshot")
    parser.add_argument("--season", type=int, default=current_cfb_season())
    parser.add_argument(
        "--output", type=Path, default=PROCESSED_DIR / "line_snapshots.parquet"
    )
    parser.add_argument(
        "--refresh-features", action="store_true",
        help="Also refresh the current line consensus, feature matrix, and shadow signals",
    )
    args = parser.parse_args()
    captured_at = datetime.now(timezone.utc)
    payload = _plain(get_lines(args.season))
    raw_path, ingestion_run_id, captured_at = save_immutable_raw_json(
        payload or [], source="cfbd_lines", season=args.season, captured_at=captured_at
    )
    snapshots = normalize_cfbd_line_snapshots(
        payload or [],
        captured_at=captured_at,
        ingestion_run_id=ingestion_run_id,
        raw_payload_path=str(raw_path.relative_to(ROOT)),
    )
    if snapshots.empty:
        print(f"No line snapshots returned for {args.season}; existing artifact unchanged")
        return 1
    destination = append_line_snapshots(snapshots, args.output)
    print(
        f"Appended {len(snapshots):,} quotes to {destination} "
        f"(run={ingestion_run_id}, raw={raw_path.relative_to(ROOT)})"
    )
    if args.refresh_features:
        save_raw_json(payload, f"lines_{args.season}")
        from utils.fetch_historical import _build_lines
        from utils.feature_engine import build_feature_matrix
        from scripts.export_shadow_totals import main as export_shadow_totals

        _build_lines(True)
        build_feature_matrix(force=True)
        export_shadow_totals()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
