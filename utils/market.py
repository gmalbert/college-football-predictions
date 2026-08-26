"""Canonical odds, market snapshots, edge calculations, and settlement."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

from utils.contracts import ensure_utc


class Market(str, Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


class Side(str, Enum):
    HOME = "home"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"


class BetResult(str, Enum):
    WIN = "W"
    LOSS = "L"
    PUSH = "P"
    VOID = "V"


@dataclass(frozen=True)
class SpreadEdge:
    model_home_margin: float
    home_spread: float
    market_home_margin: float
    edge_points: float
    side: Side


@dataclass(frozen=True)
class Settlement:
    result: BetResult
    profit: float
    return_amount: float


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def decimal_to_american(decimal_odds: float) -> float:
    decimal_odds = float(decimal_odds)
    if decimal_odds <= 1:
        raise ValueError("decimal odds must be greater than 1")
    profit = decimal_odds - 1.0
    return -100.0 / profit if profit < 1 else 100.0 * profit


def american_to_implied(odds: float) -> float:
    return 1.0 / american_to_decimal(odds)


def implied_to_american(probability: float) -> float:
    probability = float(probability)
    if not 0 < probability < 1:
        raise ValueError("probability must be strictly between 0 and 1")
    return decimal_to_american(1.0 / probability)


def remove_vig(odds: Iterable[float]) -> np.ndarray:
    """Multiplicatively normalize implied probabilities to sum to one."""
    implied = np.asarray([american_to_implied(value) for value in odds], dtype=float)
    total = implied.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("odds do not imply a valid market")
    return implied / total


def overround(odds: Iterable[float]) -> float:
    return float(sum(american_to_implied(value) for value in odds) - 1.0)


def expected_value(win_probability: float, odds: float, *, push_probability: float = 0.0) -> float:
    """Expected profit per unit staked."""
    win_probability = float(win_probability)
    push_probability = float(push_probability)
    loss_probability = 1.0 - win_probability - push_probability
    if min(win_probability, push_probability, loss_probability) < 0:
        raise ValueError("win, push, and loss probabilities must form a valid distribution")
    net_win = american_to_decimal(odds) - 1.0
    return win_probability * net_win - loss_probability


def spread_edge(model_home_margin: float, home_spread: float) -> SpreadEdge:
    """Compare a home-margin forecast to a home-team sportsbook spread.

    A book quote of home -7 corresponds to a market-implied home margin of +7.
    This explicit conversion prevents the sign inversion that previously made
    nearly every favorite appear to be a large model edge.
    """
    model_margin = float(model_home_margin)
    spread = float(home_spread)
    market_margin = -spread
    edge = model_margin - market_margin
    return SpreadEdge(
        model_home_margin=model_margin,
        home_spread=spread,
        market_home_margin=market_margin,
        edge_points=edge,
        side=Side.HOME if edge >= 0 else Side.AWAY,
    )


def normal_cover_probability(
    model_home_margin: float,
    home_spread: float,
    residual_std: float,
) -> float:
    if residual_std <= 0:
        raise ValueError("residual_std must be positive")
    threshold = -float(home_spread)
    z = (threshold - float(model_home_margin)) / float(residual_std)
    return 1.0 - NormalDist().cdf(z)


def settle_bet(
    *,
    market: Market | str,
    side: Side | str,
    home_score: float | None,
    away_score: float | None,
    line: float | None = None,
    odds: float = -110,
    stake: float = 1.0,
    void: bool = False,
) -> Settlement:
    """Settle moneyline, spread, or total bets with explicit push handling."""
    market = Market(market)
    side = Side(side)
    stake = float(stake)
    if stake < 0:
        raise ValueError("stake cannot be negative")
    if void or home_score is None or away_score is None:
        return Settlement(BetResult.VOID, 0.0, stake)
    home_score = float(home_score)
    away_score = float(away_score)

    if market == Market.MONEYLINE:
        if home_score == away_score:
            result = BetResult.PUSH
        else:
            home_won = home_score > away_score
            result = BetResult.WIN if (side == Side.HOME) == home_won else BetResult.LOSS
    elif market == Market.SPREAD:
        if line is None or side not in {Side.HOME, Side.AWAY}:
            raise ValueError("spread settlement requires a home/away side and line")
        side_margin = home_score - away_score if side == Side.HOME else away_score - home_score
        settled = side_margin + float(line)
        result = BetResult.WIN if settled > 0 else BetResult.LOSS if settled < 0 else BetResult.PUSH
    else:
        if line is None or side not in {Side.OVER, Side.UNDER}:
            raise ValueError("total settlement requires an over/under side and line")
        total = home_score + away_score
        settled = total - float(line)
        if settled == 0:
            result = BetResult.PUSH
        elif side == Side.OVER:
            result = BetResult.WIN if settled > 0 else BetResult.LOSS
        else:
            result = BetResult.WIN if settled < 0 else BetResult.LOSS

    if result == BetResult.WIN:
        profit = stake * (american_to_decimal(odds) - 1.0)
        return Settlement(result, profit, stake + profit)
    if result == BetResult.LOSS:
        return Settlement(result, -stake, 0.0)
    return Settlement(result, 0.0, stake)


def select_quotes_asof(
    quotes: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    *,
    captured_at: str = "captured_at",
) -> pd.DataFrame:
    """Select the latest quote per book/market/side at or before a cutoff."""
    required = {"game_id", "sportsbook", "market", "side", captured_at}
    missing = sorted(required.difference(quotes.columns))
    if missing:
        raise KeyError(f"quotes are missing columns: {missing}")
    frame = quotes.copy()
    frame[captured_at] = ensure_utc(frame[captured_at])
    cutoff_ts = pd.Timestamp(cutoff)
    cutoff_ts = cutoff_ts.tz_localize("UTC") if cutoff_ts.tzinfo is None else cutoff_ts.tz_convert("UTC")
    frame = frame[frame[captured_at] <= cutoff_ts]
    keys = ["game_id", "sportsbook", "market", "side"]
    return (
        frame.sort_values(captured_at)
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )


def consensus_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a snapshot to market consensus and disagreement features."""
    required = {"game_id", "market", "side", "sportsbook", "odds"}
    missing = sorted(required.difference(quotes.columns))
    if missing:
        raise KeyError(f"quotes are missing columns: {missing}")
    frame = quotes.copy()
    frame["odds"] = pd.to_numeric(frame["odds"], errors="coerce")
    if "line" not in frame.columns:
        frame["line"] = np.nan
    frame["line"] = pd.to_numeric(frame["line"], errors="coerce")
    grouped = frame.groupby(["game_id", "market", "side"], dropna=False)
    result = grouped.agg(
        consensus_line=("line", "median"),
        line_low=("line", "min"),
        line_high=("line", "max"),
        consensus_odds=("odds", "median"),
        best_odds=("odds", "max"),
        book_count=("sportsbook", "nunique"),
    ).reset_index()
    result["line_range"] = result["line_high"] - result["line_low"]
    return result


def line_movement_features(
    quotes: pd.DataFrame,
    *,
    captured_at: str = "captured_at",
) -> pd.DataFrame:
    """Compute open/current movement, volatility, age, and book dispersion."""
    required = {"game_id", "market", "side", "line", "sportsbook", captured_at}
    missing = sorted(required.difference(quotes.columns))
    if missing:
        raise KeyError(f"quotes are missing columns: {missing}")
    frame = quotes.copy()
    frame[captured_at] = ensure_utc(frame[captured_at])
    frame["line"] = pd.to_numeric(frame["line"], errors="coerce")
    keys = ["game_id", "market", "side"]
    rows = []
    for key, group in frame.sort_values(captured_at).groupby(keys, dropna=False):
        valid = group.dropna(subset=["line", captured_at])
        if valid.empty:
            continue
        latest_time = valid[captured_at].max()
        latest = valid[valid[captured_at] == latest_time]
        open_line = float(valid.iloc[0]["line"])
        current_line = float(latest["line"].median())
        rows.append(
            {
                **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                "open_line": open_line,
                "current_line": current_line,
                "line_move": current_line - open_line,
                "line_volatility": float(valid["line"].std(ddof=0)),
                "book_dispersion": float(latest["line"].max() - latest["line"].min()),
                "quote_count": int(len(valid)),
                "sportsbook_count": int(valid["sportsbook"].nunique()),
                "latest_quote_at": latest_time,
            }
        )
    return pd.DataFrame(rows)


def closing_line_value(
    *,
    market: Market | str,
    taken_line: float | None = None,
    closing_line: float | None = None,
    taken_odds: float | None = None,
    closing_odds: float | None = None,
) -> float:
    """Return bettor-positive CLV in points (spread/total) or decimal-price %."""
    market = Market(market)
    if market == Market.MONEYLINE:
        if taken_odds is None or closing_odds is None:
            return float("nan")
        return american_to_decimal(taken_odds) / american_to_decimal(closing_odds) - 1.0
    if taken_line is None or closing_line is None:
        return float("nan")
    return float(taken_line) - float(closing_line)
