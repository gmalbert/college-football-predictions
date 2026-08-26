"""Export upcoming, model-scored NCAAF bets for the Sports Picks Grid."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from statistics import NormalDist
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.betting import (  # noqa: E402
    Confidence,
    generate_moneyline_pick,
    generate_spread_pick,
    generate_total_pick,
)
from utils.market import expected_value, normal_cover_probability  # noqa: E402
from utils.models import MODEL_VERSION, load_metrics, models_trained, predict_batch  # noqa: E402
from utils.seasons import current_cfb_season  # noqa: E402

SPORT = "NCAAF"
OUT_PATH = ROOT / "data_files" / "best_bets_today.json"
LOOKAHEAD_DAYS = 6


def _write(bets: list[dict], notes: str = "") -> None:
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": str(current_cfb_season(date.today())),
            "prediction_contract": "latest available line snapshot at export time",
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(OUT_PATH)
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _tier(confidence: Confidence) -> str:
    return {
        Confidence.STRONG: "Elite",
        Confidence.MODERATE: "Strong",
        Confidence.LEAN: "Good",
        Confidence.NONE: "No Bet",
    }[confidence]


def _record(
    row: pd.Series,
    recommendation,
    *,
    line: float | None,
    odds: float,
    probability: float,
) -> dict:
    start = pd.to_datetime(row["start_date"], errors="coerce", utc=True)
    return {
        "game_id": int(row["game_id"]) if pd.notna(row.get("game_id")) else None,
        "game_date": start.date().isoformat(),
        "game_time": start.isoformat(),
        "game": f"{row['away_team']} @ {row['home_team']}",
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "bet_type": recommendation.bet_type.title(),
        "pick": recommendation.pick,
        "probability": round(float(probability), 6),
        "edge": round(float(recommendation.edge), 4),
        "tier": _tier(recommendation.confidence),
        "odds": odds,
        "line": line,
        "league": SPORT,
    }


def main() -> None:
    today = date.today()
    if today.month not in {8, 9, 10, 11, 12, 1}:
        _write([], "NCAAF off-season")
        return
    feature_path = ROOT / "data_files" / "features" / "feature_matrix.parquet"
    if not feature_path.exists() or not models_trained():
        _write([], "Feature matrix or trained models are unavailable")
        return

    metrics = load_metrics()
    decision = metrics.get("release_decision", {})
    if decision.get("decision") != "promote":
        failed = ", ".join(decision.get("failed_gates", [])) or "release gates unavailable"
        _write([], f"Model release is on hold: {failed}")
        return

    frame = pd.read_parquet(feature_path).drop_duplicates("game_id", keep="last")
    if "start_date" not in frame.columns:
        _write([], "Feature matrix has no start_date")
        return
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce", utc=True)
    end = today + timedelta(days=LOOKAHEAD_DAYS)
    game_dates = frame["start_date"].dt.date
    upcoming = frame[(game_dates >= today) & (game_dates <= end)].copy()
    if upcoming.empty:
        _write([], f"No NCAAF games in next {LOOKAHEAD_DAYS} days")
        return
    upcoming = predict_batch(upcoming)

    spread_std = float(metrics.get("spread_model", {}).get("residual_std", 0) or 0)
    total_std = float(metrics.get("total_model", {}).get("residual_std", 0) or 0)

    bets: list[dict] = []
    for _, row in upcoming.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        if pd.notna(row.get("predicted_spread")) and pd.notna(row.get("market_spread")):
            recommendation = generate_spread_pick(
                home, away, float(row["predicted_spread"]), float(row["market_spread"]),
                game_id=int(row["game_id"]),
            )
            home_cover = normal_cover_probability(
                float(row["predicted_spread"]), float(row["market_spread"]), spread_std
            ) if spread_std > 0 else 0.5
            probability = home_cover if recommendation.pick.startswith(f"Take {home} ") else 1 - home_cover
            if (
                recommendation.confidence in {Confidence.MODERATE, Confidence.STRONG}
                and expected_value(probability, -110) >= 0.02
            ):
                bets.append(
                    _record(
                        row, recommendation, line=float(row["market_spread"]), odds=-110.0,
                        probability=probability,
                    )
                )
        if pd.notna(row.get("predicted_total")) and pd.notna(row.get("market_total")):
            over_probability = (
                1 - NormalDist(float(row["predicted_total"]), total_std).cdf(float(row["market_total"]))
                if total_std > 0 else 0.5
            )
            model_over = float(row["predicted_total"]) > float(row["market_total"])
            probability = over_probability if model_over else 1 - over_probability
            recommendation = generate_total_pick(
                home, away, float(row["predicted_total"]), float(row["market_total"]),
                game_id=int(row["game_id"]), win_prob=probability,
            )
            if (
                recommendation.confidence in {Confidence.MODERATE, Confidence.STRONG}
                and expected_value(probability, -110) >= 0.02
            ):
                bets.append(
                    _record(
                        row, recommendation, line=float(row["market_total"]), odds=-110.0,
                        probability=probability,
                    )
                )
        if all(pd.notna(row.get(column)) for column in ("win_prob", "home_moneyline", "away_moneyline")):
            recommendation = generate_moneyline_pick(
                home,
                away,
                float(row["win_prob"]),
                float(row["home_moneyline"]),
                float(row["away_moneyline"]),
                game_id=int(row["game_id"]),
            )
            if recommendation is not None:
                odds = float(row["home_moneyline"]) if home in recommendation.pick else float(row["away_moneyline"])
                if expected_value(recommendation.win_prob, odds) >= 0.02:
                    bets.append(
                        _record(
                            row, recommendation, line=None, odds=odds,
                            probability=recommendation.win_prob,
                        )
                    )
    bets.sort(key=lambda bet: (bet["tier"] != "Elite", -float(bet["edge"])))
    _write(bets, "" if bets else "No bets met release thresholds")


if __name__ == "__main__":
    main()
