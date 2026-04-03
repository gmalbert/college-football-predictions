# Betting Strategy Roadmap

This document covers the quantitative betting strategy the platform will
implement — from edge detection through bankroll management.

---

## 1. Core Philosophy

The site is **not** a tout service. It's a tool that:

1. Produces independent model-derived lines.
2. Compares them to sportsbook lines.
3. Surfaces games where the model disagrees by a meaningful amount (the **edge**).
4. Tracks long-term accuracy transparently.

---

## 2. Edge Detection

### What is an edge?

```
Edge = Model Line − Book Line
```

For spreads:  
- Model says Home −3.5, Book says Home −7 → Edge = +3.5 points in favor of
  the home team's opponent (take the dog or the points).

For totals:  
- Model says 56, Book says 52 → Edge = +4 → lean Over.

### Confidence Tiers

| Tier | Spread Edge | Total Edge | Suggested Action |
|------|-------------|------------|------------------|
| 🔥 **Strong** | ≥ 3.0 pts | ≥ 4.0 pts | Featured pick |
| ✅ **Moderate** | 2.0–2.9 pts | 2.5–3.9 pts | Standard bet |
| ⚠️ **Lean** | 1.0–1.9 pts | 1.5–2.4 pts | Informational only |
| ⚪ **No edge** | < 1.0 pts | < 1.5 pts | Skip |

```python
"""betting.py — Edge detection and bet classification."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class Confidence(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    LEAN = "lean"
    NONE = "none"

@dataclass
class BetRecommendation:
    game_id: int
    home_team: str
    away_team: str
    bet_type: str           # "spread", "total", "moneyline"
    model_line: float
    book_line: float
    edge: float
    confidence: Confidence
    pick: str               # e.g. "Take Michigan +7" or "Over 52"

def classify_spread_edge(edge: float) -> Confidence:
    abs_edge = abs(edge)
    if abs_edge >= 3.0:
        return Confidence.STRONG
    elif abs_edge >= 2.0:
        return Confidence.MODERATE
    elif abs_edge >= 1.0:
        return Confidence.LEAN
    return Confidence.NONE

def classify_total_edge(edge: float) -> Confidence:
    abs_edge = abs(edge)
    if abs_edge >= 4.0:
        return Confidence.STRONG
    elif abs_edge >= 2.5:
        return Confidence.MODERATE
    elif abs_edge >= 1.5:
        return Confidence.LEAN
    return Confidence.NONE

def generate_spread_pick(
    home: str, away: str, model_spread: float, book_spread: float
) -> BetRecommendation:
    """
    model_spread and book_spread are from the home team's perspective
    (negative = home favored).
    """
    edge = model_spread - book_spread
    confidence = classify_spread_edge(edge)

    if edge > 0:
        # Model says home is less favored than book → take home / points
        pick = f"Take {home} {book_spread:+.1f}"
    else:
        pick = f"Take {away} +{abs(book_spread):.1f}"

    return BetRecommendation(
        game_id=0,
        home_team=home,
        away_team=away,
        bet_type="spread",
        model_line=model_spread,
        book_line=book_spread,
        edge=edge,
        confidence=confidence,
        pick=pick,
    )

def generate_total_pick(
    home: str, away: str, model_total: float, book_total: float
) -> BetRecommendation:
    edge = model_total - book_total
    confidence = classify_total_edge(edge)
    pick = f"Over {book_total:.1f}" if edge > 0 else f"Under {book_total:.1f}"

    return BetRecommendation(
        game_id=0,
        home_team=home,
        away_team=away,
        bet_type="total",
        model_line=model_total,
        book_line=book_total,
        edge=edge,
        confidence=confidence,
        pick=pick,
    )
```

---

## 3. Bankroll Management

### 3a. Flat Betting

Simplest approach — wager the same fixed amount on every qualified bet.

```python
def flat_bet_size(bankroll: float, unit_pct: float = 0.02) -> float:
    """Return bet size as a fixed percentage of bankroll."""
    return bankroll * unit_pct
```

- **Typical unit:** 1–3% of bankroll.
- **Pro:** Simple, limits variance.
- **Con:** Doesn't scale bet size with perceived edge.

### 3b. Kelly Criterion

Optimal growth strategy — bet proportionally to your edge.

```python
import numpy as np

def kelly_fraction(win_prob: float, odds: float) -> float:
    """
    Kelly fraction for a bet with given win probability and decimal odds.

    Parameters
    ----------
    win_prob : float
        Model's estimated probability the bet wins.
    odds : float
        Decimal odds (e.g., 1.91 for -110).

    Returns
    -------
    float
        Fraction of bankroll to wager (0 if negative EV).
    """
    b = odds - 1  # net payout per dollar wagered
    q = 1 - win_prob
    f = (b * win_prob - q) / b
    return max(0.0, f)

def fractional_kelly(win_prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Quarter-Kelly (default) — reduces variance while capturing most of the
    growth benefit.
    """
    return kelly_fraction(win_prob, odds) * fraction
```

**Why quarter-Kelly?** Full Kelly is theoretically optimal but assumes
perfect probability estimates. Since our model is imperfect, fractional
Kelly reduces the risk of ruin substantially.

### 3c. Bankroll Simulator

```python
"""bankroll_sim.py — Simulate bankroll trajectory over a season."""
import numpy as np
import pandas as pd

def simulate_season(
    bets: pd.DataFrame,
    starting_bankroll: float = 1000.0,
    strategy: str = "flat",  # "flat" or "kelly"
    unit_pct: float = 0.02,
    kelly_fraction: float = 0.25,
) -> pd.DataFrame:
    """
    Simulate bankroll growth over a series of bets.

    bets DataFrame must have columns:
        - win_prob: model's win probability for the pick
        - decimal_odds: payout odds
        - result: 1 if bet won, 0 if lost
    """
    bankroll = starting_bankroll
    history = []

    for _, bet in bets.iterrows():
        if strategy == "flat":
            wager = bankroll * unit_pct
        else:
            f = fractional_kelly(bet["win_prob"], bet["decimal_odds"], kelly_fraction)
            wager = bankroll * f

        wager = min(wager, bankroll)  # can't bet more than you have

        if bet["result"] == 1:
            profit = wager * (bet["decimal_odds"] - 1)
        else:
            profit = -wager

        bankroll += profit
        history.append({
            "bankroll": bankroll,
            "wager": wager,
            "profit": profit,
            "cumulative_profit": bankroll - starting_bankroll,
        })

    return pd.DataFrame(history)
```

---

## 4. Expected Value (EV) Calculation

```python
def expected_value(win_prob: float, decimal_odds: float) -> float:
    """
    EV per dollar wagered.
    Positive = profitable long-term.
    """
    return win_prob * (decimal_odds - 1) - (1 - win_prob)

def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return 1 + american / 100
    return 1 + 100 / abs(american)

def implied_probability(american: int) -> float:
    """Convert American odds to implied probability (no-vig)."""
    if american < 0:
        return abs(american) / (abs(american) + 100)
    return 100 / (american + 100)
```

---

## 5. Break-Even Thresholds

At standard −110 juice:

| Bet Type | Break-Even Win % |
|----------|-----------------|
| Spread (−110) | 52.38% |
| Moneyline (varies) | = implied prob from odds |
| Total (−110) | 52.38% |

A model that hits **53–55% ATS** over a full season is genuinely profitable
after juice.

---

## 6. Bet Tracking & Reporting

Track every bet the model recommends so you can audit:

```python
"""bet_tracker.py — Log and summarize betting performance."""
import pandas as pd
from pathlib import Path

BET_LOG = Path("data_files/bet_log.csv")

def log_bet(
    season: int, week: int, game_id: int,
    bet_type: str, pick: str, odds: int,
    wager: float, result: int | None = None,
) -> None:
    """Append a bet to the persistent log."""
    new_row = pd.DataFrame([{
        "season": season, "week": week, "game_id": game_id,
        "bet_type": bet_type, "pick": pick, "odds": odds,
        "wager": wager, "result": result,
    }])
    if BET_LOG.exists():
        existing = pd.read_csv(BET_LOG)
        df = pd.concat([existing, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(BET_LOG, index=False)

def season_summary(season: int) -> dict:
    """Summarize season betting performance."""
    df = pd.read_csv(BET_LOG)
    df = df[df["season"] == season].dropna(subset=["result"])

    wins = (df["result"] == 1).sum()
    losses = (df["result"] == 0).sum()
    total_wagered = df["wager"].sum()

    profit = df.apply(
        lambda r: r["wager"] * (american_to_decimal(int(r["odds"])) - 1)
        if r["result"] == 1 else -r["wager"],
        axis=1,
    ).sum()

    return {
        "record": f"{wins}-{losses}",
        "win_pct": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "total_wagered": total_wagered,
        "profit": profit,
        "roi": profit / total_wagered if total_wagered > 0 else 0,
    }
```

---

## 7. Responsible Gambling Notice

Every page should include (in the footer or sidebar):

> ⚠️ **Disclaimer:** This site provides model-based predictions for
> informational and entertainment purposes only. It is not financial advice.
> Gamble responsibly. If you or someone you know has a gambling problem,
> call **1-800-GAMBLER**.
