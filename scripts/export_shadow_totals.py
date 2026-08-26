"""Export closing-time total signals for prospective validation, never wagering."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.models import load_metrics, models_trained, predict_batch  # noqa: E402
from utils.prospective_ledger import append_shadow_signals, select_quote_as_of  # noqa: E402
from utils.release import load_current_release  # noqa: E402

OUT_PATH = ROOT / "data_files" / "shadow_total_signals.json"
MAX_HOURS_TO_KICKOFF = 12


def _write(signals: list[dict], note: str, *, metadata: dict | None = None) -> None:
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "deployment_status": "shadow_not_actionable",
            "prediction_contract": "final eligible pre-kickoff snapshot",
            "maximum_hours_to_kickoff": MAX_HOURS_TO_KICKOFF,
            "note": note,
            **(metadata or {}),
        },
        "signals": signals,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(OUT_PATH)
    print(f"[NCAAF shadow totals] Wrote {len(signals)} signals -> {OUT_PATH}")


def main() -> None:
    if not models_trained():
        _write([], "Required model artifacts are unavailable")
        return
    metrics = load_metrics()
    model_metrics = metrics.get("total_cover_model", {})
    release = model_metrics.get("strategy_release", {})
    if release.get("status") not in {"shadow", "promote"}:
        _write([], "Total-side model has not passed retrospective gates")
        return
    feature_path = ROOT / "data_files" / "features" / "feature_matrix.parquet"
    if not feature_path.exists():
        _write([], "Feature matrix is unavailable")
        return

    now = pd.Timestamp.now(tz="UTC")
    cutoff = now + pd.Timedelta(hours=MAX_HOURS_TO_KICKOFF)
    frame = pd.read_parquet(feature_path).drop_duplicates("game_id", keep="last")
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce", utc=True)
    eligible = frame[
        frame["start_date"].between(now, cutoff, inclusive="both")
        & frame["total_points"].isna()
        & frame["market_total"].notna()
    ].copy()
    if eligible.empty:
        _write([], "No priced games are within the closing-time window")
        return

    scored = predict_batch(eligible)
    snapshots_path = ROOT / "data_files" / "processed" / "line_snapshots.parquet"
    snapshots = pd.read_parquet(snapshots_path) if snapshots_path.exists() else pd.DataFrame()
    threshold = float(
        model_metrics.get("strategy", {}).get("probability_edge_threshold", 0.075)
    )
    eligible_sides = set(release.get("eligible_sides", []))
    scored["probability_edge"] = (scored["total_over_prob"] - 0.5).abs()
    scored = scored[
        scored["total_over_prob"].notna()
        & (scored["probability_edge"] >= threshold)
    ]
    signals: list[dict] = []
    release = load_current_release()
    for _, row in scored.iterrows():
        over = float(row["total_over_prob"]) > 0.5
        side = "over" if over else "under"
        if side not in eligible_sides:
            continue
        quote = select_quote_as_of(
            snapshots,
            game_id=int(row["game_id"]),
            market="total",
            side=side,
            as_of=now,
        )
        if quote is None or pd.isna(quote.get("line")):
            continue
        signals.append(
            {
                "game_id": int(row["game_id"]),
                "game": f"{row['away_team']} @ {row['home_team']}",
                "kickoff": row["start_date"].isoformat(),
                "side": side,
                "market_total": float(row["market_total"]),
                "prediction_time": now.isoformat(),
                "sportsbook": str(quote["sportsbook"]),
                "taken_line": float(quote["line"]),
                "taken_odds": float(quote["odds"]) if pd.notna(quote.get("odds")) else None,
                "source_snapshot_at": pd.Timestamp(quote["captured_at"]).isoformat(),
                "ingestion_run_id": quote.get("ingestion_run_id"),
                "over_probability": round(float(row["total_over_prob"]), 6),
                "selected_probability": round(
                    float(row["total_over_prob"] if over else 1 - row["total_over_prob"]), 6
                ),
                "probability_edge": round(float(row["probability_edge"]), 6),
                "opening_total": (
                    float(row["market_total_open"])
                    if pd.notna(row.get("market_total_open")) else None
                ),
                "line_move": (
                    float(row["market_total_move"])
                    if pd.notna(row.get("market_total_move")) else None
                ),
                "book_count": (
                    int(row["market_total_book_count"])
                    if pd.notna(row.get("market_total_book_count")) else 0
                ),
                "status": "shadow_not_a_bet",
            }
        )
    metadata = {
        "release_id": release.get("release_id"),
        "model_version": metrics.get("model_version"),
        "strategy_version": "total-over-v2.2-locked",
    }
    append_shadow_signals(signals, metadata=metadata)
    _write(
        signals,
        "Prospective validation only; prices and CLV must pass before promotion",
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
