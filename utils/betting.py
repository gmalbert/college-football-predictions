"""utils/betting.py

Edge detection, bet recommendation generation, and Kelly-criterion sizing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Confidence(Enum):
    STRONG   = "strong"
    MODERATE = "moderate"
    LEAN     = "lean"
    NONE     = "none"


@dataclass
class BetRecommendation:
    game_id: int | None
    home_team: str
    away_team: str
    bet_type: str            # "spread" | "total" | "moneyline"
    model_line: float
    book_line: float
    edge: float
    confidence: Confidence
    pick: str                # human-readable recommendation
    win_prob: float = 0.5    # model-implied win probability for the pick


CONFIDENCE_EMOJI: dict[Confidence, str] = {
    Confidence.STRONG:   "🔥",
    Confidence.MODERATE: "✅",
    Confidence.LEAN:     "⚠️",
    Confidence.NONE:     "⚪",
}

CONFIDENCE_LABEL: dict[Confidence, str] = {
    Confidence.STRONG:   "STRONG",
    Confidence.MODERATE: "MODERATE",
    Confidence.LEAN:     "LEAN",
    Confidence.NONE:     "—",
}


# ─────────────────────────────── classifiers ─────────────────────────────────

def classify_spread_edge(edge: float) -> Confidence:
    a = abs(edge)
    if a >= 3.0:
        return Confidence.STRONG
    if a >= 2.0:
        return Confidence.MODERATE
    if a >= 1.0:
        return Confidence.LEAN
    return Confidence.NONE


def classify_total_edge(edge: float) -> Confidence:
    a = abs(edge)
    if a >= 4.0:
        return Confidence.STRONG
    if a >= 2.5:
        return Confidence.MODERATE
    if a >= 1.5:
        return Confidence.LEAN
    return Confidence.NONE


# ─────────────────────────────── generators ──────────────────────────────────

def generate_spread_pick(
    home: str,
    away: str,
    model_spread: float,
    book_spread: float,
    game_id: int | None = None,
    win_prob: float = 0.5,
) -> BetRecommendation:
    """
    model_spread and book_spread are from the home team's perspective
    (negative = home favored, e.g., home −7).

    Edge > 0 → model thinks home is less favored → take home / more points.
    Edge < 0 → model thinks away is less favored → take away / more points.
    """
    edge = model_spread - book_spread
    conf = classify_spread_edge(edge)

    if edge > 0:
        pick = f"Take {home} {book_spread:+.1f}"
    else:
        away_book_spread = -book_spread
        pick = f"Take {away} {away_book_spread:+.1f}"

    return BetRecommendation(
        game_id=game_id, home_team=home, away_team=away,
        bet_type="spread", model_line=model_spread, book_line=book_spread,
        edge=abs(edge), confidence=conf, pick=pick, win_prob=win_prob,
    )


def generate_total_pick(
    home: str,
    away: str,
    model_total: float,
    book_total: float,
    game_id: int | None = None,
) -> BetRecommendation:
    """Positive edge → model expects more scoring → Over."""
    edge = model_total - book_total
    conf = classify_total_edge(edge)
    direction = "Over" if edge > 0 else "Under"
    pick = f"{direction} {book_total:.1f}"

    return BetRecommendation(
        game_id=game_id, home_team=home, away_team=away,
        bet_type="total", model_line=model_total, book_line=book_total,
        edge=abs(edge), confidence=conf, pick=pick,
    )


def generate_moneyline_pick(
    home: str,
    away: str,
    win_prob: float,
    home_ml: float,
    away_ml: float,
    game_id: int | None = None,
) -> BetRecommendation | None:
    """
    Return a moneyline recommendation if the model's implied win probability
    exceeds the book's implied probability by a meaningful margin.
    home_ml / away_ml: American odds (e.g., -150, +130).
    """
    book_home_prob = _american_to_implied(home_ml)
    book_away_prob = _american_to_implied(away_ml)

    home_edge = win_prob - book_home_prob
    away_edge = (1 - win_prob) - book_away_prob

    if home_edge >= 0.04:
        conf = Confidence.STRONG if home_edge >= 0.08 else Confidence.MODERATE
        return BetRecommendation(
            game_id=game_id, home_team=home, away_team=away,
            bet_type="moneyline", model_line=win_prob,
            book_line=book_home_prob, edge=home_edge,
            confidence=conf, pick=f"{home} ML ({home_ml:+.0f})",
            win_prob=win_prob,
        )
    if away_edge >= 0.04:
        conf = Confidence.STRONG if away_edge >= 0.08 else Confidence.MODERATE
        return BetRecommendation(
            game_id=game_id, home_team=home, away_team=away,
            bet_type="moneyline", model_line=1 - win_prob,
            book_line=book_away_prob, edge=away_edge,
            confidence=conf, pick=f"{away} ML ({away_ml:+.0f})",
            win_prob=1 - win_prob,
        )
    return None


# ─────────────────────────────── bankroll ────────────────────────────────────

def kelly_fraction(win_prob: float, odds: float = -110) -> float:
    """
    Full Kelly criterion fraction of bankroll.
    odds: American odds (default -110 = standard -vig spread bet).
    Returns value capped at 25% of bankroll.
    """
    b = 100 / abs(odds) if odds < 0 else odds / 100
    q = 1.0 - win_prob
    f = (b * win_prob - q) / b
    return max(0.0, min(f, 0.25))


def half_kelly(win_prob: float, odds: float = -110) -> float:
    """Half Kelly (more conservative; typical for sports betting)."""
    return kelly_fraction(win_prob, odds) / 2


def simulate_bankroll(
    bets: list[dict],        # [{"result": "W"|"L"|"P", "odds": -110, "stake": 1}, ...]
    starting: float = 1000,
) -> list[float]:
    """
    Simulate cumulative bankroll over a list of bet results.
    Returns list of bankroll values (length = len(bets) + 1, first = starting).
    """
    bankroll = [starting]
    current = starting
    for bet in bets:
        odds    = bet.get("odds", -110)
        stake   = bet.get("stake", 1.0) * current
        result  = bet.get("result", "L")
        if result == "W":
            payout = stake * (100 / abs(odds) if odds < 0 else odds / 100)
            current += payout
        elif result == "L":
            current -= stake
        bankroll.append(current)
    return bankroll


# ─────────────────────────────── utility ─────────────────────────────────────

def _american_to_implied(odds: float) -> float:
    """Convert American odds to implied win probability (no vig removed)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)
