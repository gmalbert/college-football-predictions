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
    BetRecommendation,
    Confidence,
    generate_spread_pick,
    generate_total_pick,
)
from utils.market import (  # noqa: E402
    american_to_implied,
    expected_value,
    normal_cover_probability,
)
from utils.models import MODEL_VERSION, load_metrics, models_trained, predict_batch  # noqa: E402
from utils.seasons import current_cfb_season  # noqa: E402

SPORT = "NCAAF"
OUT_PATH = ROOT / "data_files" / "best_bets_today.json"
LOOKAHEAD_DAYS = 6
SNAPSHOT_PATH = ROOT / "data_files" / "processed" / "line_snapshots.parquet"
MAX_QUOTE_AGE = pd.Timedelta(hours=24)
MAX_PAIR_AGE = pd.Timedelta(minutes=30)


def _write(
    bets: list[dict],
    notes: str = "",
    *,
    release_decision: dict | None = None,
) -> None:
    release_decision = release_decision or {}
    release_status = release_decision.get("decision", "unknown")
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "tier_definition": "edge-v1",
            "edge_definition": "pick probability minus median no-vig probability from paired sportsbook quotes",
            "season": str(current_cfb_season(date.today())),
            "prediction_contract": "latest available line snapshot at export time",
            "release_status": release_status,
            "publication_mode": (
                "validated" if release_status == "promote" else "provisional_research"
            ),
        },
        "bets": bets,
    }
    failed_gates = release_decision.get("failed_gates", [])
    if failed_gates:
        payload["meta"]["release_failed_gates"] = failed_gates
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(OUT_PATH)
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _tier(edge: float) -> str:
    for tier, threshold in (("Elite", .06), ("Strong", .03), ("Good", .01)):
        if edge >= threshold:
            return tier
    return "Standard"


def _load_current_quotes() -> pd.DataFrame:
    """Load fresh, pregame sportsbook snapshots; ParlayAPI remains shadow-only."""
    required = {
        "game_id", "sportsbook", "market", "side", "line", "odds",
        "captured_at", "source",
    }
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame()
    try:
        quotes = pd.read_parquet(SNAPSHOT_PATH)
    except (OSError, ValueError):
        return pd.DataFrame()
    if not required.issubset(quotes.columns):
        return pd.DataFrame()

    quotes = quotes.copy()
    quotes["game_id"] = pd.to_numeric(quotes["game_id"], errors="coerce")
    quotes["line"] = pd.to_numeric(quotes["line"], errors="coerce")
    quotes["odds"] = pd.to_numeric(quotes["odds"], errors="coerce")
    quotes["captured_at"] = pd.to_datetime(quotes["captured_at"], utc=True, errors="coerce")
    quotes["source"] = quotes["source"].astype("string").fillna("unknown").str.casefold()
    quotes = quotes.dropna(subset=["game_id", "captured_at", "odds"])
    quotes = quotes[quotes["odds"].ne(0)]
    quotes = quotes[~quotes["source"].isin({"parlay", "parlay_api"})]

    now = pd.Timestamp.now(tz="UTC")
    quotes = quotes[quotes["captured_at"].between(now - MAX_QUOTE_AGE, now)]
    if "is_live" in quotes.columns:
        live = quotes["is_live"].astype("string").str.casefold().isin({"1", "true", "yes", "y"})
        quotes = quotes[~live]

    if quotes.empty:
        return quotes
    keys = ["game_id", "source", "sportsbook", "market", "side"]
    return (
        quotes.sort_values("captured_at")
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )


def _price_for_pick(
    quotes: pd.DataFrame,
    *,
    game_id: int,
    market: str,
    side: str,
    line: float | None,
) -> dict | None:
    """Price a pick from paired, same-book quotes at the selected line."""
    if quotes.empty:
        return None
    game_quotes = quotes[
        quotes["game_id"].eq(game_id) & quotes["market"].eq(market)
    ]
    opposite = {"home": "away", "away": "home", "over": "under", "under": "over"}[side]
    pairs: list[dict] = []

    for (source, sportsbook), book in game_quotes.groupby(
        ["source", "sportsbook"], dropna=False, sort=False
    ):
        chosen = book[book["side"].eq(side)]
        other = book[book["side"].eq(opposite)]
        if line is not None:
            opposite_line = -line if market == "spread" else line
            chosen = chosen[chosen["line"].sub(line).abs().le(0.01)]
            other = other[other["line"].sub(opposite_line).abs().le(0.01)]
        if chosen.empty or other.empty:
            continue

        chosen_quote = chosen.sort_values("captured_at").iloc[-1]
        other_quote = other.sort_values("captured_at").iloc[-1]
        if abs(chosen_quote["captured_at"] - other_quote["captured_at"]) > MAX_PAIR_AGE:
            continue
        try:
            picked_odds = float(chosen_quote["odds"])
            other_odds = float(other_quote["odds"])
            picked_implied = american_to_implied(picked_odds)
            other_implied = american_to_implied(other_odds)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        total = picked_implied + other_implied
        if not 0 < total:
            continue
        pairs.append(
            {
                "market_probability": picked_implied / total,
                "odds": picked_odds,
                "source": str(source),
                "sportsbook": str(sportsbook),
                "captured_at": chosen_quote["captured_at"],
            }
        )

    if not pairs:
        return None
    best_price = max(pairs, key=lambda quote: quote["odds"])
    return {
        **best_price,
        "market_probability": float(pd.Series(
            [quote["market_probability"] for quote in pairs], dtype=float
        ).median()),
        "paired_book_count": len(pairs),
    }


def _record(
    row: pd.Series,
    recommendation,
    *,
    line: float | None,
    probability: float,
    market_quote: dict,
) -> dict:
    start = pd.to_datetime(row["start_date"], errors="coerce", utc=True)
    market_probability = float(market_quote["market_probability"])
    edge = float(probability) - market_probability
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
        "confidence": round(float(probability), 6),
        "market_probability": round(market_probability, 6),
        "edge": round(edge, 4),
        "tier": _tier(edge),
        "source_edge": round(float(recommendation.edge), 4),
        "source_edge_unit": "probability" if recommendation.bet_type == "moneyline" else "points",
        "odds": float(market_quote["odds"]),
        "sportsbook": market_quote["sportsbook"],
        "odds_source": market_quote["source"],
        "price_captured_at": market_quote["captured_at"].isoformat(),
        "market_quote_count": int(market_quote["paired_book_count"]),
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
    release_on_hold = decision.get("decision") != "promote"
    failed = ", ".join(decision.get("failed_gates", [])) or "release gates unavailable"

    frame = pd.read_parquet(feature_path).drop_duplicates("game_id", keep="last")
    if "start_date" not in frame.columns:
        _write([], "Feature matrix has no start_date")
        return
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce", utc=True)
    end = today + timedelta(days=LOOKAHEAD_DAYS)
    game_dates = frame["start_date"].dt.date
    upcoming = frame[(game_dates >= today) & (game_dates <= end)].copy()
    if upcoming.empty:
        _write(
            [],
            f"No NCAAF games in next {LOOKAHEAD_DAYS} days",
            release_decision=decision,
        )
        return
    upcoming = predict_batch(upcoming)

    spread_std = float(metrics.get("spread_model", {}).get("residual_std", 0) or 0)
    total_std = float(metrics.get("total_model", {}).get("residual_std", 0) or 0)

    quotes = _load_current_quotes()
    bets: list[dict] = []
    for _, row in upcoming.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        game_id = int(row["game_id"])

        if pd.notna(row.get("predicted_spread")) and pd.notna(row.get("market_spread")):
            recommendation = generate_spread_pick(
                home, away, float(row["predicted_spread"]), float(row["market_spread"]),
                game_id=game_id,
            )
            home_cover = normal_cover_probability(
                float(row["predicted_spread"]), float(row["market_spread"]), spread_std
            ) if spread_std > 0 else 0.5
            is_home_pick = recommendation.pick.startswith(f"Take {home} ")
            side = "home" if is_home_pick else "away"
            probability = home_cover if is_home_pick else 1 - home_cover
            selected_line = (
                float(row["market_spread"])
                if is_home_pick else -float(row["market_spread"])
            )
            market_quote = _price_for_pick(
                quotes, game_id=game_id, market="spread", side=side, line=selected_line
            )
            if (
                market_quote is not None
                and probability - market_quote["market_probability"] >= 0.02
                and expected_value(probability, market_quote["odds"]) >= 0.02
            ):
                bets.append(
                    _record(
                        row, recommendation, line=selected_line,
                        probability=probability, market_quote=market_quote,
                    )
                )

        if pd.notna(row.get("predicted_total")) and pd.notna(row.get("market_total")):
            over_probability = (
                1 - NormalDist(float(row["predicted_total"]), total_std).cdf(float(row["market_total"]))
                if total_std > 0 else 0.5
            )
            model_over = float(row["predicted_total"]) > float(row["market_total"])
            side = "over" if model_over else "under"
            probability = over_probability if model_over else 1 - over_probability
            recommendation = generate_total_pick(
                home, away, float(row["predicted_total"]), float(row["market_total"]),
                game_id=game_id, win_prob=probability,
            )
            market_quote = _price_for_pick(
                quotes, game_id=game_id, market="total", side=side,
                line=float(row["market_total"]),
            )
            if (
                market_quote is not None
                and probability - market_quote["market_probability"] >= 0.02
                and expected_value(probability, market_quote["odds"]) >= 0.02
            ):
                bets.append(
                    _record(
                        row, recommendation, line=float(row["market_total"]),
                        probability=probability, market_quote=market_quote,
                    )
                )

        if pd.notna(row.get("win_prob")):
            try:
                home_probability = float(row["win_prob"])
            except (TypeError, ValueError):
                home_probability = float("nan")
            if not 0 < home_probability < 1:
                continue
            home_quote = _price_for_pick(
                quotes, game_id=game_id, market="moneyline", side="home", line=None
            )
            away_quote = _price_for_pick(
                quotes, game_id=game_id, market="moneyline", side="away", line=None
            )
            if home_quote is None or away_quote is None:
                continue

            candidates = []
            for side, team, probability, quote in (
                ("home", home, home_probability, home_quote),
                ("away", away, 1 - home_probability, away_quote),
            ):
                edge = probability - quote["market_probability"]
                if edge >= 0.04 and expected_value(probability, quote["odds"]) >= 0.02:
                    candidates.append((edge, side, team, probability, quote))
            if candidates:
                edge, side, team, probability, quote = max(candidates, key=lambda item: item[0])
                confidence = Confidence.STRONG if edge >= 0.08 else Confidence.MODERATE
                recommendation = BetRecommendation(
                    game_id=game_id,
                    home_team=home,
                    away_team=away,
                    bet_type="moneyline",
                    model_line=probability,
                    book_line=quote["market_probability"],
                    edge=edge,
                    confidence=confidence,
                    pick=f"{team} ML ({quote['odds']:+.0f})",
                    win_prob=probability,
                )
                bets.append(
                    _record(
                        row, recommendation, line=None,
                        probability=probability, market_quote=quote,
                    )
                )
    bets.sort(key=lambda bet: (bet["tier"] != "Elite", -float(bet["edge"])))
    if bets and release_on_hold:
        notes = f"Published provisionally while model release is on hold: {failed}"
    elif bets:
        notes = ""
    else:
        notes = "No bets met export thresholds"
        if quotes.empty:
            notes += "; no paired non-shadow market quotes from the last 24 hours"
        if release_on_hold:
            notes += f"; model release remains on hold: {failed}"
    _write(bets, notes, release_decision=decision)


if __name__ == "__main__":
    main()
