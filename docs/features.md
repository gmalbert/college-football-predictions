# Feature Engineering Roadmap

**Status:** ✅ Completed — implemented in `utils/feature_engine.py` and reflected in model training.

This document defines every feature (and feature family) the models will
consume, along with code to compute them from raw CFBD / ESPN data.

---

## 1. Feature Categories Overview

**Status:** ✅ Completed

| Category | # Features | Source |
|----------|-----------|--------|
| Elo & Power Ratings | 6 | Derived / CFBD ratings |
| Offensive Efficiency | 12 | CFBD advanced stats |
| Defensive Efficiency | 12 | CFBD advanced stats |
| Special Teams | 4 | CFBD stats |
| Turnover & Penalty | 4 | CFBD game stats |
| Recruiting & Talent | 4 | CFBD recruiting |
| Schedule & Context | 8 | CFBD games / ESPN |
| Betting Market | 6 | CFBD lines / Odds API |
| Weather | 4 | Weather API (future) |
| **Total** | **~60** | |

---

## 2. Detailed Feature Definitions

**Status:** ✅ Completed

### 2a. Elo & Power Ratings

**Status:** ✅ Completed

| Feature | Description |
|---------|-------------|
| `elo_home` | Home team Elo entering the game |
| `elo_away` | Away team Elo entering the game |
| `elo_diff` | `elo_home - elo_away` |
| `sp_plus_home` | Home team S&P+ rating |
| `sp_plus_away` | Away team S&P+ rating |
| `sp_plus_diff` | `sp_plus_home - sp_plus_away` |

### 2b. Offensive Efficiency (rolling 5-game average)

**Status:** ✅ Completed

| Feature | Description |
|---------|-------------|
| `off_epa_per_play` | Offensive EPA per play |
| `off_success_rate` | % of plays with positive EPA |
| `off_explosiveness` | Average EPA on successful plays |
| `off_ppa` | Predicted points added per play |
| `off_yards_per_play` | Total yards / total plays |
| `off_passing_epa` | Passing EPA per dropback |
| `off_rushing_epa` | Rushing EPA per carry |
| `off_third_down_pct` | 3rd-down conversion rate |
| `off_red_zone_pct` | Red-zone scoring rate |
| `off_scoring_ops_pct` | Points per scoring opportunity |
| `off_havoc_allowed` | Havoc plays allowed on offense |
| `off_pace` | Plays per game (tempo proxy) |

### 2c. Defensive Efficiency (rolling 5-game average)

**Status:** ✅ Completed

Mirror of offensive features with `def_` prefix, measured from the
opponent's perspective:

| Feature | Description |
|---------|-------------|
| `def_epa_per_play` | Defensive EPA per play (lower = better) |
| `def_success_rate` | Opponent success rate allowed |
| `def_explosiveness` | Opponent avg EPA on successful plays |
| `def_ppa` | Def PPA per play |
| `def_yards_per_play` | Yards allowed per play |
| `def_passing_epa` | Pass defense EPA |
| `def_rushing_epa` | Rush defense EPA |
| `def_third_down_pct` | Opponent 3rd-down conversion rate |
| `def_red_zone_pct` | Opponent red-zone scoring rate |
| `def_havoc_rate` | Havoc rate generated on defense |
| `def_sack_rate` | Sacks per opponent dropback |
| `def_turnover_rate` | Turnovers forced per game |

### 2d. Special Teams

**Status:** ✅ Completed

| Feature | Description |
|---------|-------------|
| `st_kick_return_avg` | Average kick return yards |
| `st_punt_return_avg` | Average punt return yards |
| `st_fg_pct` | Field goal percentage |
| `st_punt_net_avg` | Net punting average |

### 2e. Turnover & Penalties

**Status:** ✅ Completed

| Feature | Description |
|---------|-------------|
| `turnover_margin_season` | Season turnover margin |
| `turnover_margin_l5` | Last 5 games turnover margin |
| `penalty_yards_per_game` | Penalty yards per game |
| `penalty_count_per_game` | Penalties per game |

### 2f. Recruiting & Talent

**Status:** ✅ Completed

| Feature | Description |
|---------|-------------|
| `recruiting_rank` | Team's composite recruiting rank (3-yr avg) |
| `talent_composite` | CFBD talent composite score |
| `blue_chip_ratio` | % of 4★ and 5★ recruits |
| `recruiting_diff` | `home_recruiting - away_recruiting` |

### 2g. Schedule & Context

**Status:** ✅ Completed

| Feature | Description |
|---------|-------------|
| `home_flag` | 1 = home, 0 = away, 0.5 = neutral site |
| `conference_game` | 1 if conference matchup |
| `rivalry_flag` | 1 if rivalry game |
| `rest_days_home` | Days since last game (home) |
| `rest_days_away` | Days since last game (away) |
| `rest_advantage` | `rest_days_home - rest_days_away` |
| `week` | Season week number |
| `season` | Year |

### 2h. Betting Market Features

**Status:** ✅ Completed

| Feature | Description |
|---------|-------------|
| `market_spread` | Opening sportsbook spread |
| `market_total` | Opening O/U line |
| `market_moneyline_home` | Home moneyline odds |
| `market_moneyline_away` | Away moneyline odds |
| `spread_movement` | Close − Open spread |
| `total_movement` | Close − Open total |

### 2i. Weather (Future)

| Feature | Description |
|---------|-------------|
| `weather_temp` | Temperature (°F) at kickoff |
| `weather_wind` | Wind speed (mph) |
| `weather_precip_prob` | Precipitation probability |
| `is_dome` | 1 if indoor/dome venue |

---

## 3. Feature Engineering Code

**Status:** ✅ Completed

```python
"""features.py — Build feature matrix from raw data."""
import pandas as pd
import numpy as np


def rolling_stats(df: pd.DataFrame, team_col: str, stat_cols: list[str],
                  window: int = 5) -> pd.DataFrame:
    """
    Compute rolling-window averages for each team, ensuring no data leakage
    (rolling window excludes the current game).
    """
    df = df.sort_values(["season", "week"])
    result_frames = []

    for team, group in df.groupby(team_col):
        rolled = (
            group[stat_cols]
            .shift(1)               # exclude current game
            .rolling(window, min_periods=1)
            .mean()
        )
        rolled.columns = [f"{c}_l{window}" for c in stat_cols]
        rolled[team_col] = team
        rolled.index = group.index
        result_frames.append(rolled)

    return pd.concat(result_frames).sort_index()


def compute_elo_features(games: pd.DataFrame, elo_model) -> pd.DataFrame:
    """Run Elo model forward through the schedule, producing per-game Elo diffs."""
    records = []
    for _, row in games.iterrows():
        home, away = row["home_team"], row["away_team"]
        records.append({
            "game_id": row["id"],
            "elo_home": elo_model.get_rating(home),
            "elo_away": elo_model.get_rating(away),
            "elo_diff": elo_model.get_rating(home) - elo_model.get_rating(away),
        })
        if pd.notna(row.get("home_points")):
            home_won = row["home_points"] > row["away_points"]
            elo_model.update(home, away, home_won)

    return pd.DataFrame(records)


def compute_recruiting_features(recruiting: pd.DataFrame,
                                 games: pd.DataFrame) -> pd.DataFrame:
    """
    Join 3-year rolling recruiting composite onto the game-level dataframe.
    """
    # Average last 3 years of recruiting
    recruiting = recruiting.sort_values(["team", "year"])
    recruiting["talent_3yr"] = (
        recruiting.groupby("team")["points"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    latest = recruiting.drop_duplicates("team", keep="last")[["team", "talent_3yr"]]
    games = games.merge(latest.rename(columns={"team": "home_team", "talent_3yr": "recruiting_home"}),
                        on="home_team", how="left")
    games = games.merge(latest.rename(columns={"team": "away_team", "talent_3yr": "recruiting_away"}),
                        on="away_team", how="left")
    games["recruiting_diff"] = games["recruiting_home"] - games["recruiting_away"]
    return games


def compute_rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Calculate days of rest for home and away teams."""
    games = games.sort_values(["season", "week"])

    for side in ["home", "away"]:
        col = f"{side}_team"
        games[f"prev_game_date_{side}"] = (
            games.groupby(col)["start_date"].shift(1)
        )
        games[f"rest_days_{side}"] = (
            pd.to_datetime(games["start_date"]) -
            pd.to_datetime(games[f"prev_game_date_{side}"])
        ).dt.days
        games.drop(columns=[f"prev_game_date_{side}"], inplace=True)

    games["rest_advantage"] = games["rest_days_home"] - games["rest_days_away"]
    return games


def build_feature_matrix(games: pd.DataFrame,
                          team_stats: pd.DataFrame,
                          advanced: pd.DataFrame,
                          recruiting: pd.DataFrame,
                          lines: pd.DataFrame,
                          elo_model) -> pd.DataFrame:
    """
    Master function — combines all feature sources into one model-ready
    DataFrame.
    """
    # 1. Elo
    elo_df = compute_elo_features(games, elo_model)
    df = games.merge(elo_df, on="game_id", how="left")

    # 2. Rolling offensive / defensive stats
    off_cols = ["off_epa_per_play", "off_success_rate", "off_yards_per_play"]
    def_cols = ["def_epa_per_play", "def_success_rate", "def_yards_per_play"]

    for side, cols in [("home", off_cols + def_cols), ("away", off_cols + def_cols)]:
        rolled = rolling_stats(team_stats, f"{side}_team", cols, window=5)
        df = df.merge(rolled, left_index=True, right_index=True, how="left", suffixes=("", f"_{side}"))

    # 3. Recruiting
    df = compute_recruiting_features(recruiting, df)

    # 4. Rest days
    df = compute_rest_days(df)

    # 5. Betting lines
    if lines is not None and len(lines):
        df = df.merge(
            lines[["game_id", "spread", "over_under"]].rename(
                columns={"spread": "market_spread", "over_under": "market_total"}
            ),
            on="game_id",
            how="left",
        )

    # 6. Context flags
    df["home_flag"] = df.get("neutral_site", False).apply(lambda x: 0.5 if x else 1.0)
    df["conference_game"] = (df["home_conference"] == df["away_conference"]).astype(int)

    # 7. Target variables
    df["home_margin"] = df["home_points"] - df["away_points"]
    df["total_points"] = df["home_points"] + df["away_points"]
    df["home_win"] = (df["home_margin"] > 0).astype(int)

    return df
```

---

## 4. Feature Importance Tracking

After training, log feature importances so you can prune weak features:

```python
import pandas as pd

def get_xgb_importances(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract and sort XGBoost feature importance."""
    scores = model.get_score(importance_type="gain")
    df = pd.DataFrame(
        [(feature_names[int(k.replace("f", ""))], v) for k, v in scores.items()],
        columns=["feature", "gain"],
    ).sort_values("gain", ascending=False)
    return df
```

---

## 5. Feature Pipeline Checklist

- [ ] Pull raw game data (`/games`) → store as Parquet
- [ ] Pull advanced stats (`/stats/season/advanced`) → store as Parquet
- [ ] Pull betting lines (`/lines`) → store as Parquet
- [ ] Pull recruiting (`/recruiting/teams`) → store as Parquet
- [ ] Pull ratings (`/ratings/sp`, `/ratings/elo`) → store as Parquet
- [ ] Run Elo forward pass → compute `elo_diff`
- [ ] Compute rolling window stats (no leakage!)
- [ ] Merge all features into single DataFrame
- [ ] Validate: no NaN targets, no future data in features
- [ ] Save feature matrix to `data_files/features/`
