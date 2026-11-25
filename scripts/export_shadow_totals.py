"""Export closing-time total signals for prospective validation, never wagering."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.export_best_bets import _load_current_quotes, _price_for_pick  # noqa: E402
from utils.market import expected_value  # noqa: E402
from utils.models import MODEL_VERSION, load_metrics, models_trained, predict_batch  # noqa: E402
from utils.prospective_ledger import PROSPECTIVE_STRATEGIES, append_shadow_signals  # noqa: E402
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
    if metrics.get("model_version") != MODEL_VERSION:
        _write([], "The prospective strategy is waiting for matching model artifacts")
        return
    model_metrics = metrics.get("total_cover_model", {})
    strategy_release = model_metrics.get("strategy_release", {})
    eligible_sides = set(strategy_release.get("eligible_sides", []))
    if "over" not in eligible_sides:
        _write([], "No totals research side met the retrospective selection criteria")
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
        frame["start_date"].between(now, cutoff, inclusive="right")
        & frame["total_points"].isna()
        & frame["market_total"].notna()
    ].copy()
    if eligible.empty:
        _write([], "No priced games are within the closing-time window")
        return

    scored = predict_batch(eligible)
    snapshots = _load_current_quotes(now)
    threshold = float(
        model_metrics.get("strategy", {}).get("probability_edge_threshold", 0.075)
    )
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
        strategy = PROSPECTIVE_STRATEGIES.get(("total", side))
        if side not in eligible_sides or strategy is None:
            continue
        line = float(row["market_total"])
        quote = _price_for_pick(
            snapshots,
            game_id=int(row["game_id"]),
            market="total",
            side=side,
            line=line,
        )
        probability = float(row["total_over_prob"] if over else 1 - row["total_over_prob"])
        if quote is None or expected_value(probability, quote["odds"]) < 0.02:
            continue
        signals.append(
            {
                "game_id": int(row["game_id"]),
                "game": f"{row['away_team']} @ {row['home_team']}",
                "kickoff": row["start_date"].isoformat(),
                "side": side,
                "market": "total",
                "strategy_version": strategy,
                "market_total": line,
                "prediction_time": now.isoformat(),
                "sportsbook": str(quote["sportsbook"]),
                "taken_line": line,
                "taken_odds": quote["odds"],
                "source_snapshot_at": pd.Timestamp(quote["captured_at"]).isoformat(),
                "odds_source": quote["source"],
                "ingestion_run_id": quote.get("ingestion_run_id"),
                "over_probability": round(float(row["total_over_prob"]), 6),
                "selected_probability": round(probability, 6),
                "model_probability": round(probability, 6),
                "market_probability": quote["market_probability"],
                "expected_value": expected_value(probability, quote["odds"]),
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
        "strategy_version": PROSPECTIVE_STRATEGIES[("total", "over")],
    }
    append_shadow_signals(signals, metadata=metadata)
    _write(
        signals,
        "Prospective research only. Both retrospective and closing-price criteria must pass.",
        metadata={**metadata, "strategy_release_status": strategy_release.get("status", "hold")},
    )


if __name__ == "__main__":
    main()
