"""Capture a forward-only CFBD odds snapshot for movement and CLV tracking."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.cfbd_client import get_lines  # noqa: E402
from utils.odds_api import get_ncaaf_odds, is_configured, to_cfbd_line_payload  # noqa: E402
from utils.odds_api_io import (  # noqa: E402
    get_ncaaf_odds as get_odds_api_io_odds,
    is_configured as odds_api_io_is_configured,
    to_cfbd_line_payload as odds_api_io_to_cfbd_line_payload,
)
from utils.rundown_client import (  # noqa: E402
    get_ncaaf_events, is_configured as rundown_is_configured,
    to_cfbd_line_payload as rundown_to_cfbd_line_payload,
)
from utils.odds_ingestion import append_line_snapshots, normalize_cfbd_line_snapshots  # noqa: E402
from utils.seasons import current_cfb_season  # noqa: E402
from utils.storage import PROCESSED_DIR, save_raw_json, save_immutable_raw_json  # noqa: E402
import pandas as pd


def _plain(value):
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Append the latest sportsbook snapshot")
    parser.add_argument("--season", type=int, default=current_cfb_season())
    parser.add_argument(
        "--output", type=Path, default=PROCESSED_DIR / "line_snapshots.parquet"
    )
    parser.add_argument(
        "--refresh-features", action="store_true",
        help="Also refresh the current line consensus, feature matrix, and shadow signals",
    )
    parser.add_argument(
        "--source", choices=("auto", "odds-api-io", "rundown", "odds", "cfbd"), default="auto",
        help="Market provider (default: OddsAPI.io, TheRundown, Odds API, then CFBD)",
    )
    args = parser.parse_args()
    captured_at = datetime.now(timezone.utc)
    source = "cfbd_lines"
    payload = []
    use_odds_api_io = args.source == "odds-api-io" or (
        args.source == "auto" and odds_api_io_is_configured()
    )
    use_rundown = args.source == "rundown" or (
        args.source == "auto" and not use_odds_api_io and rundown_is_configured()
    )
    use_odds = args.source == "odds" or (
        args.source == "auto" and not use_odds_api_io and not use_rundown and is_configured()
    )
    if use_odds_api_io:
        events = get_odds_api_io_odds()
        games_path = PROCESSED_DIR / "games.parquet"
        games = pd.read_parquet(games_path) if games_path.exists() else pd.DataFrame()
        payload = odds_api_io_to_cfbd_line_payload(events, games, season=args.season)
        source = "odds_api_io"
        if not payload:
            print("No matched OddsAPI.io quotes returned; existing artifact unchanged")
            return 1
    if use_rundown:
        events = get_ncaaf_events()
        games_path = PROCESSED_DIR / "games.parquet"
        games = pd.read_parquet(games_path) if games_path.exists() else pd.DataFrame()
        payload = rundown_to_cfbd_line_payload(events, games, season=args.season)
        source = "therundown"
        if not payload:
            print("No matched TheRundown quotes returned; existing artifact unchanged")
            return 1
    if use_odds:
        odds_events = get_ncaaf_odds()
        games_path = PROCESSED_DIR / "games.parquet"
        games = pd.read_parquet(games_path) if games_path.exists() else pd.DataFrame()
        payload = to_cfbd_line_payload(odds_events, games, season=args.season)
        source = "odds_api"
        if not payload:
            print("No matched Odds API quotes returned; existing artifact unchanged")
            return 1
    if args.source == "cfbd" or (args.source == "auto" and not use_odds_api_io and not use_rundown and not use_odds):
        payload = _plain(get_lines(args.season))
        source = "cfbd_lines"
    raw_path, ingestion_run_id, captured_at = save_immutable_raw_json(
        payload or [], source=source, season=args.season, captured_at=captured_at
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
