"""Export upcoming, model-scored NCAAF bets for the Sports Picks Grid."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from math import isfinite
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.betting import (  # noqa: E402
    BetRecommendation,
    Confidence,
    generate_spread_pick,
)
from utils.market import (  # noqa: E402
    american_to_decimal,
    american_to_implied,
    expected_value,
    normal_cover_probability,
)
from utils.models import MODEL_VERSION, load_metrics, models_trained, predict_batch  # noqa: E402
from utils.odds_ingestion import recorded_provider_matches  # noqa: E402
from utils.prospective_ledger import PROSPECTIVE_STRATEGIES, append_shadow_signals  # noqa: E402
from utils.release import load_current_release  # noqa: E402
from utils.seasons import current_cfb_season  # noqa: E402

SPORT = "NCAAF"
OUT_PATH = ROOT / "data_files" / "best_bets_today.json"
LOOKAHEAD_DAYS = 6
SNAPSHOT_PATH = ROOT / "data_files" / "processed" / "line_snapshots.parquet"
MAX_QUOTE_AGE = pd.Timedelta(hours=24)
MAX_PAIR_AGE = pd.Timedelta(minutes=30)


def _release_is_promoted(decision: dict) -> bool:
    return decision.get("decision") == "promote" and not decision.get("failed_gates")


def _write(
    bets: list[dict],
    notes: str = "",
    *,
    release_decision: dict,
    market_statuses: dict,
) -> None:
    promoted = _release_is_promoted(release_decision)
    release_status = "promote" if promoted else "hold"
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "tier_definition": "edge-v1",
            "edge_definition": "pick probability minus displayed price break-even probability",
            "market_edge_definition": "pick probability minus median no-vig probability across paired sportsbooks",
            "market_quote_count_definition": "number of sportsbooks with both sides paired",
            "price_contract": "same sportsbook and provider, exact line, pregame quotes within 24 hours and no more than 30 minutes apart",
            "market_release_status": market_statuses,
            "totals_publication": "research_only",
            "season": str(current_cfb_season(date.today())),
            "prediction_contract": "latest available line snapshot at export time",
            "release_status": release_status,
            "publication_mode": (
                "validated" if promoted else "provisional_research"
            ),
        },
        "bets": bets if promoted else [],
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
    print(f"[{SPORT}] Wrote {len(payload['bets'])} bets -> {OUT_PATH}")


def _tier(edge: float) -> str:
    for tier, threshold in (("Elite", .06), ("Strong", .03), ("Good", .01)):
        if edge >= threshold:
            return tier
    return "Standard"


def _load_current_quotes(now: pd.Timestamp) -> pd.DataFrame:
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

    quotes = quotes.loc[recorded_provider_matches(quotes)].copy()
    quotes["game_id"] = pd.to_numeric(quotes["game_id"], errors="coerce")
    quotes["line"] = pd.to_numeric(quotes["line"], errors="coerce")
    quotes["odds"] = pd.to_numeric(quotes["odds"], errors="coerce")
    quotes["captured_at"] = pd.to_datetime(quotes["captured_at"], utc=True, errors="coerce")
    quotes["source"] = quotes["source"].astype("string").str.strip().str.casefold()
    quotes["sportsbook"] = quotes["sportsbook"].astype("string").str.strip()
    quotes = quotes.dropna(subset=["game_id", "source", "sportsbook", "captured_at", "odds"])
    quotes = quotes[quotes["source"].ne("") & quotes["sportsbook"].ne("")]
    # Snapshot odds are American prices; values strictly between -100 and +100
    # are malformed (and would create false implied probabilities).
    valid_american = quotes["odds"].le(-100) | quotes["odds"].ge(100)
    quotes = quotes[valid_american & quotes["odds"].map(isfinite)]
    quotes = quotes[~quotes["source"].isin({"parlay", "parlay_api"})]

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
    pairs_by_book: dict[str, dict] = {}

    for (source, sportsbook), book in game_quotes.groupby(
        ["source", "sportsbook"], dropna=False, sort=False
    ):
        chosen = book[book["side"].eq(side)]
        other = book[book["side"].eq(opposite)]
        if line is not None:
            opposite_line = -line if market == "spread" else line
            chosen = chosen[chosen["line"].eq(line)]
            other = other[other["line"].eq(opposite_line)]
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
        pair = {
            "market_probability": picked_implied / total,
            "price_implied_probability": picked_implied,
            "odds": picked_odds,
            "opposing_odds": other_odds,
            "source": str(source),
            "sportsbook": str(sportsbook),
            "captured_at": chosen_quote["captured_at"],
            "opposing_captured_at": other_quote["captured_at"],
            "ingestion_run_id": chosen_quote.get("ingestion_run_id"),
        }
        # A sportsbook can arrive through multiple providers. Count it once.
        book_key = str(sportsbook).casefold()
        previous_pair = pairs_by_book.get(book_key)
        if previous_pair is None or max(pair["captured_at"], pair["opposing_captured_at"]) > max(
            previous_pair["captured_at"], previous_pair["opposing_captured_at"]
        ):
            pairs_by_book[book_key] = pair

    if not pairs_by_book:
        return None
    pairs = list(pairs_by_book.values())
    best_price = max(pairs, key=lambda quote: american_to_decimal(quote["odds"]))
    return {
        **best_price,
        "selected_book_no_vig_probability": best_price["market_probability"],
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
    side: str,
    probability: float,
    market_quote: dict,
) -> dict:
    start = pd.to_datetime(row["start_date"], errors="coerce", utc=True)
    market_probability = float(market_quote["market_probability"])
    price_implied = market_quote["price_implied_probability"]
    edge = round(probability - price_implied, 4)
    market_edge = probability - market_probability
    return {
        "game_id": int(row["game_id"]) if pd.notna(row.get("game_id")) else None,
        "game_date": start.date().isoformat(),
        "game_time": start.isoformat(),
        "game": f"{row['away_team']} @ {row['home_team']}",
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "bet_type": recommendation.bet_type.title(),
        "side": side,
        "pick": recommendation.pick,
        "probability": round(float(probability), 6),
        "confidence": round(float(probability), 6),
        "market_probability": round(market_probability, 6),
        "price_implied_probability": round(price_implied, 6),
        "selected_book_no_vig_probability": round(market_quote["selected_book_no_vig_probability"], 6),
        "edge": edge,
        "price_edge": edge,
        "market_edge": round(market_edge, 4),
        "market_edge_display": f"{market_edge * 100:.1f} pp",
        "expected_value_per_unit": round(expected_value(probability, market_quote["odds"]), 4),
        "edge_unit": "probability",
        "edge_display": f"{edge * 100:.1f} pp",
        "tier": _tier(edge),
        "source_edge": round(float(recommendation.edge), 4),
        "source_edge_unit": "probability" if recommendation.bet_type == "moneyline" else "points",
        "odds": float(market_quote["odds"]),
        "sportsbook": market_quote["sportsbook"],
        "odds_source": market_quote["source"],
        "ingestion_run_id": (
            str(market_quote["ingestion_run_id"])
            if pd.notna(market_quote["ingestion_run_id"]) else None
        ),
        "price_captured_at": market_quote["captured_at"].isoformat(),
        "opposing_odds": market_quote["opposing_odds"],
        "opposing_price_captured_at": market_quote["opposing_captured_at"].isoformat(),
        "market_quote_count": int(market_quote["paired_book_count"]),
        "line": line,
        "league": SPORT,
    }

def _record_prospective_picks(bets: list[dict], now: pd.Timestamp) -> None:
    release_id = load_current_release().get("release_id")
    signals = []
    for bet in bets:
        market = bet["bet_type"].lower()
        signals.append({
            "game_id": bet["game_id"],
            "market": market,
            "side": bet["side"],
            "prediction_time": now,
            "kickoff": bet["game_time"],
            "sportsbook": bet["sportsbook"],
            "taken_line": bet["line"],
            "taken_odds": bet["odds"],
            "source_snapshot_at": bet["price_captured_at"],
            "odds_source": bet["odds_source"],
            "ingestion_run_id": bet["ingestion_run_id"],
            "model_probability": bet["probability"],
            "market_probability": bet["market_probability"],
            "expected_value": bet["expected_value_per_unit"],
            "strategy_version": PROSPECTIVE_STRATEGIES[(market, None)],
        })
    append_shadow_signals(signals, metadata={"model_version": MODEL_VERSION, "release_id": release_id})


def main(*, record_prospective: bool = False) -> None:
    now = pd.Timestamp.now(tz="UTC")
    today = now.date()
    metrics = load_metrics()
    decision = metrics.get("release_decision", {})
    export_metadata = {
        "release_decision": decision,
        "market_statuses": metrics.get("market_release_status", {}),
    }
    approved = _release_is_promoted(decision)
    collect = record_prospective and metrics.get("model_version") == MODEL_VERSION
    failed = ", ".join(decision.get("failed_gates", [])) or "release approval unavailable"
    hold_note = f"RESEARCH ONLY. Public picks withheld while the release is on hold ({failed})."
    if not approved and not collect:
        _write([], hold_note, **export_metadata)
        return
    if today.month not in {8, 9, 10, 11, 12, 1}:
        _write([], "NCAAF off-season", **export_metadata)
        return
    feature_path = ROOT / "data_files" / "features" / "feature_matrix.parquet"
    if not feature_path.exists() or not models_trained():
        _write([], "Feature matrix or trained models are unavailable", **export_metadata)
        return

    frame = pd.read_parquet(feature_path).drop_duplicates("game_id", keep="last")
    if "start_date" not in frame.columns:
        _write([], "Feature matrix has no start_date", **export_metadata)
        return
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce", utc=True)
    end = today + timedelta(days=LOOKAHEAD_DAYS)
    game_dates = frame["start_date"].dt.date
    upcoming = frame[(frame["start_date"] > now) & (game_dates <= end)].copy()
    if upcoming.empty:
        _write(
            [],
            f"No NCAAF games in next {LOOKAHEAD_DAYS} days",
            **export_metadata,
        )
        return
    spread_metrics = metrics.get("spread_model", {})
    spread_std = float(spread_metrics.get("priced_residual_std", spread_metrics.get("residual_std")) or 0)

    quotes = _load_current_quotes(now)
    quotes_by_game = {
        game_id: group for game_id, group in quotes.groupby("game_id", sort=False)
    } if not quotes.empty else {}
    moneyline_prices = {}
    for game_id in upcoming["game_id"]:
        game_quotes = quotes_by_game.get(game_id)
        if game_quotes is None:
            continue
        home_quote = _price_for_pick(game_quotes, game_id=int(game_id), market="moneyline", side="home", line=None)
        away_quote = _price_for_pick(game_quotes, game_id=int(game_id), market="moneyline", side="away", line=None)
        if home_quote is not None and away_quote is not None:
            moneyline_prices[game_id] = (home_quote, away_quote)

    # Score and compare moneyline prices using one as-of market. A cached
    # feature built from older snapshots can otherwise create a phantom edge.
    fresh_market = upcoming["game_id"].map({
        game_id: home_quote["market_probability"]
        for game_id, (home_quote, _) in moneyline_prices.items()
    })
    upcoming["market_home_prob"] = fresh_market.fillna(upcoming.get("market_home_prob", fresh_market))
    upcoming = predict_batch(upcoming)
    bets: list[dict] = []
    # Totals remain in the closing-time shadow feed. The points regressor
    # has no validated export-time cover probabilities.
    for _, row in upcoming.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        game_id = int(row["game_id"])

        if spread_std > 0 and pd.notna(row.get("predicted_spread")) and pd.notna(row.get("market_spread")):
            recommendation = generate_spread_pick(
                home, away, float(row["predicted_spread"]), float(row["market_spread"]),
                game_id=game_id,
            )
            home_cover = normal_cover_probability(
                float(row["predicted_spread"]), float(row["market_spread"]), spread_std
            )
            is_home_pick = float(row["predicted_spread"]) + float(row["market_spread"]) > 0
            side = "home" if is_home_pick else "away"
            probability = home_cover if is_home_pick else 1 - home_cover
            selected_line = (
                float(row["market_spread"])
                if is_home_pick else -float(row["market_spread"])
            )
            market_quote = _price_for_pick(
                quotes_by_game.get(game_id, quotes.iloc[:0]),
                game_id=game_id, market="spread", side=side, line=selected_line
            )
            if (
                market_quote is not None
                and probability - market_quote["market_probability"] >= 0.02
                and expected_value(probability, market_quote["odds"]) >= 0.02
            ):
                bets.append(
                    _record(
                        row, recommendation, line=selected_line, side=side,
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
            if game_id not in moneyline_prices:
                continue
            home_quote, away_quote = moneyline_prices[game_id]

            candidates = []
            for side, team, probability, quote in (
                ("home", home, home_probability, home_quote),
                ("away", away, 1 - home_probability, away_quote),
            ):
                edge = probability - quote["market_probability"]
                if edge >= 0.04 and expected_value(probability, quote["odds"]) >= 0.02:
                    price_edge = probability - quote["price_implied_probability"]
                    candidates.append((price_edge, edge, side, team, probability, quote))
            if candidates:
                _, market_edge, side, team, probability, quote = max(candidates, key=lambda item: item[0])
                confidence = Confidence.STRONG if market_edge >= 0.08 else Confidence.MODERATE
                recommendation = BetRecommendation(
                    game_id=game_id,
                    home_team=home,
                    away_team=away,
                    bet_type="moneyline",
                    model_line=probability,
                    book_line=quote["market_probability"],
                    edge=market_edge,
                    confidence=confidence,
                    pick=f"{team} ML ({quote['odds']:+.0f})",
                    win_prob=probability,
                )
                bets.append(
                    _record(
                        row, recommendation, line=None, side=side,
                        probability=probability, market_quote=quote,
                    )
                )
    bets.sort(key=lambda bet: (-bet["edge"], bet["game_time"]))
    if collect:
        _record_prospective_picks(bets, now)
    notes = "" if bets else "No bets met export thresholds with fresh, paired pregame prices."
    _write(bets, notes if approved else hold_note, **export_metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-prospective", action="store_true", help="Record priced signals in the zero-stake validation ledger.")
    main(record_prospective=parser.parse_args().record_prospective)
