"""utils/elo.py

Custom Elo rating model for college football.
Used for real-time rating updates when CFBD pre-computed Elo is unavailable
(e.g., current-season games before CFBD posts their ratings).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_FACTOR = 20
HOME_ADVANTAGE = 65        # Elo points added to home team's effective rating
INITIAL_ELO = 1500
REVERSION_FACTOR = 0.33    # mean-revert 1/3 toward 1500 each off-season


def expected_win_prob(elo_a: float, elo_b: float) -> float:
    """Return P(team A beats team B) given their Elo ratings."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def update_elo(
    winner_elo: float,
    loser_elo: float,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """Return (new_winner_elo, new_loser_elo) after one game result."""
    exp_w = expected_win_prob(winner_elo, loser_elo)
    return (
        winner_elo + k * (1 - exp_w),
        loser_elo  + k * (0 - (1 - exp_w)),
    )


def season_revert(elo: float) -> float:
    """Mean-revert rating toward 1500 at the start of a new season."""
    return elo + REVERSION_FACTOR * (INITIAL_ELO - elo)


def margin_multiplier(margin: float, elo_difference: float) -> float:
    """Bounded margin-of-victory multiplier with favorite autocorrelation control."""
    raw = np.log1p(abs(float(margin))) * 2.2 / (abs(float(elo_difference)) * 0.001 + 2.2)
    return float(np.clip(raw, 0.5, 2.5))


def build_pregame_elo_features(
    games_df: pd.DataFrame,
    *,
    k: float = K_FACTOR,
    home_adv: float = HOME_ADVANTAGE,
    initial: float = INITIAL_ELO,
    reversion: float = REVERSION_FACTOR,
) -> pd.DataFrame:
    """Walk forward and emit ratings before each game result is observed.

    Unlike the CFBD season Elo endpoint, these values are safe for historical
    training rows because the current game and all future games are excluded.
    """
    required = {"game_id", "season", "home_team", "away_team"}
    missing = sorted(required.difference(games_df.columns))
    if missing:
        raise KeyError(f"games are missing columns: {missing}")
    order = [column for column in ("start_date", "game_id") if column in games_df.columns]
    games = games_df.sort_values(order or ["season", "week", "game_id"]).copy()
    ratings: dict[str, float] = {}
    rows: list[dict] = []
    previous_season: int | None = None

    for _, game in games.iterrows():
        season = int(game["season"])
        if previous_season is not None and season != previous_season:
            ratings = {
                team: rating + reversion * (initial - rating)
                for team, rating in ratings.items()
            }
        previous_season = season
        home = str(game["home_team"])
        away = str(game["away_team"])
        home_rating = ratings.get(home, initial)
        away_rating = ratings.get(away, initial)
        neutral = bool(game.get("neutral_site", False))
        advantage = 0.0 if neutral else home_adv
        expected = expected_win_prob(home_rating + advantage, away_rating)
        rows.append(
            {
                "game_id": game["game_id"],
                "home_elo_pregame": home_rating,
                "away_elo_pregame": away_rating,
                "elo_diff": home_rating - away_rating,
                "elo_home_win_prob": expected,
            }
        )

        home_score = pd.to_numeric(pd.Series([game.get("home_score")]), errors="coerce").iloc[0]
        away_score = pd.to_numeric(pd.Series([game.get("away_score")]), errors="coerce").iloc[0]
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        margin = float(home_score - away_score)
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        multiplier = margin_multiplier(margin, home_rating + advantage - away_rating)
        delta = k * multiplier * (actual - expected)
        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta
    return pd.DataFrame(rows)


class EloModel:
    """Track and update Elo ratings across a full multi-season dataset."""

    def __init__(self, k: float = K_FACTOR, home_adv: float = HOME_ADVANTAGE):
        self.k = k
        self.home_adv = home_adv
        self.ratings: dict[str, float] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def get_rating(self, team: str) -> float:
        return self.ratings.setdefault(team, INITIAL_ELO)

    def predict(self, home: str, away: str) -> float:
        """Return P(home team wins), accounting for home-field advantage."""
        return expected_win_prob(
            self.get_rating(home) + self.home_adv,
            self.get_rating(away),
        )

    def update(self, home: str, away: str, home_won: bool) -> None:
        """Update ratings after one game result."""
        he = self.get_rating(home) + self.home_adv
        ae = self.get_rating(away)
        if home_won:
            nw, nl = update_elo(he, ae, self.k)
        else:
            nl, nw = update_elo(ae, he, self.k)
        self.ratings[home] = nw - self.home_adv
        self.ratings[away] = nl

    def new_season(self) -> None:
        """Apply off-season mean reversion for all tracked teams."""
        for team in list(self.ratings):
            self.ratings[team] = season_revert(self.ratings[team])

    def train_on_games(self, games_df: pd.DataFrame) -> None:
        """
        Walk forward through sorted games, updating ratings game by game.

        Required columns: season, week, home_team, away_team, home_win.
        Season transitions trigger automatic mean reversion.
        """
        games = games_df.sort_values(["season", "week"]).reset_index(drop=True)
        prev_season: int | None = None
        for _, row in games.iterrows():
            if prev_season is not None and int(row["season"]) != prev_season:
                self.new_season()
            prev_season = int(row["season"])
            self.update(row["home_team"], row["away_team"], bool(row["home_win"]))

    def ratings_snapshot(self) -> pd.DataFrame:
        """Return current ratings as a sorted DataFrame."""
        return (
            pd.DataFrame(
                [{"team": t, "elo": r} for t, r in self.ratings.items()]
            )
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
