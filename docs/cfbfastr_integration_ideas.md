# cfbfastR Integration Ideas for Tailgate Edge

> **Source reviewed:** [cfbfastR](https://cfbfastr.sportsdataverse.org/) — an R package wrapping the [CollegeFootballData.com](https://collegefootballdata.com/) and ESPN APIs.  
> **GitHub:** [sportsdataverse/cfbfastR](https://github.com/sportsdataverse/cfbfastR)

This document identifies data sources, features, models, and scraping techniques from cfbfastR that are **not yet used** in this project, along with Python implementation code for each.

---

## Table of Contents

1. [New Data Sources / API Endpoints](#1-new-data-sources--api-endpoints)
   - 1a. Game Weather
   - 1b. Venues (Dome / Elevation / Capacity)
   - 1c. FPI Ratings
   - 1d. SRS (Simple Rating System) Ratings
   - 1e. Pre-Game Win Probabilities
   - 1f. PPA / WEPA (Predicted Points Added, Opponent-Adjusted)
   - 1g. Player Returning Production
   - 1h. Transfer Portal Data
   - 1i. Game Media / TV Coverage
   - 1j. Play-by-Play Data
   - 1k. Drive-Level Data
   - 1l. Player Usage Metrics
   - 1m. Game Player Stats
   - 1n. Coach History
   - 1o. Head-to-Head Matchup Records (API-native)
2. [New Features for Modeling](#2-new-features-for-modeling)
3. [New Model Ideas](#3-new-model-ideas)
   - 3a. Custom EPA Model
   - 3b. Win Probability Added (WPA) Model
   - 3c. Field Goal Expected Points Model
4. [New Pages / Visualizations](#4-new-pages--visualizations)
5. [Implementation Priority Matrix](#5-implementation-priority-matrix)

---

## 1. New Data Sources / API Endpoints

### 1a. Game Weather ✅

cfbfastR exposes `cfbd_game_weather()` returning temperature, humidity, wind speed/direction, precipitation, snowfall, pressure, and weather condition per game. **Not currently fetched by this project.**

**CFBD API endpoint:** `GET /games/weather`

**Returns:** `game_id, season, week, game_indoors, venue, temperature, dew_point, humidity, precipitation, snowfall, wind_direction, wind_speed, pressure, weather_condition_code, weather_condition`

```python
# utils/cfbd_client.py — add this function

def get_game_weather(year: int, week: int | None = None, season_type: str = "regular") -> list:
    """Get weather data for games in a given year/week."""
    api = cfbd.GamesApi(_client())
    try:
        kwargs = {"year": year, "season_type": cfbd.SeasonType(season_type)}
        if week is not None:
            kwargs["week"] = week
        return api.get_game_weather(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_game_weather error: {e}")
        return []
```

```python
# utils/storage.py — add fetch + save logic

def fetch_weather(year: int):
    """Fetch and save game weather data for a season."""
    raw = cfbd_client.get_game_weather(year)
    records = []
    for w in raw:
        records.append({
            "game_id": w.id if hasattr(w, 'id') else getattr(w, 'game_id', None),
            "season": year,
            "week": getattr(w, 'week', None),
            "game_indoors": getattr(w, 'game_indoors', None),
            "venue": getattr(w, 'venue', None),
            "temperature": getattr(w, 'temperature', None),
            "dew_point": getattr(w, 'dew_point', None),
            "humidity": getattr(w, 'humidity', None),
            "precipitation": getattr(w, 'precipitation', None),
            "snowfall": getattr(w, 'snowfall', None),
            "wind_direction": getattr(w, 'wind_direction', None),
            "wind_speed": getattr(w, 'wind_speed', None),
            "pressure": getattr(w, 'pressure', None),
            "weather_condition": getattr(w, 'weather_condition', None),
        })
    save_raw_json(records, f"weather_{year}.json")
    return records
```

**Feature engineering additions:**
```python
# utils/feature_engine.py — weather features

def _add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather data and create weather-related features."""
    wdf = weather_df.copy()
    # Binary: is the game indoors/dome?
    wdf["is_dome"] = wdf["game_indoors"].fillna(False).astype(int)
    # Adverse weather flag: wind > 15 mph OR precipitation > 0 OR temp < 35°F
    wdf["adverse_weather"] = (
        (wdf["wind_speed"].fillna(0) > 15) |
        (wdf["precipitation"].fillna(0) > 0) |
        (wdf["temperature"].fillna(60) < 35)
    ).astype(int)
    # High wind flag (affects passing game)
    wdf["high_wind"] = (wdf["wind_speed"].fillna(0) > 20).astype(int)

    merge_cols = ["game_id", "is_dome", "temperature", "wind_speed",
                  "humidity", "precipitation", "adverse_weather", "high_wind"]
    return df.merge(wdf[merge_cols], on="game_id", how="left")
```

---

### 1b. Venues (Dome / Elevation / Capacity) ✅

cfbfastR exposes `cfbd_venues()` returning dome status, capacity, grass/turf, elevation, latitude/longitude, and timezone. Useful for home-field advantage modeling (e.g., high-elevation venues like BYU, Air Force, Colorado).

**CFBD API endpoint:** `GET /venues`

```python
# utils/cfbd_client.py

def get_venues() -> list:
    """Get all college football venue information."""
    api = cfbd.VenuesApi(_client())
    try:
        return api.get_venues()
    except Exception as e:
        logger.error(f"CFBD get_venues error: {e}")
        return []
```

```python
# Feature engineering: elevation advantage
def _add_venue_features(df: pd.DataFrame, venues_df: pd.DataFrame) -> pd.DataFrame:
    """Add venue-based features like elevation, dome, capacity."""
    vdf = venues_df[["venue_id", "dome", "elevation", "capacity", "grass"]].copy()
    vdf["elevation"] = pd.to_numeric(vdf["elevation"], errors="coerce").fillna(0)
    vdf["is_dome"] = vdf["dome"].fillna(False).astype(int)
    vdf["is_grass"] = vdf["grass"].fillna(False).astype(int)
    # High altitude flag (>5000 ft) — significant impact on kicking/passing
    vdf["high_altitude"] = (vdf["elevation"] > 5000).astype(int)
    return df.merge(vdf, on="venue_id", how="left")
```

---

### 1c. FPI (Football Power Index) Ratings ✅

cfbfastR wraps `cfbd_ratings_fpi()` — ESPN's proprietary power index with overall/offense/defense/special teams efficiencies plus resume ranks (SOS, SOR, game control).

**CFBD API endpoint:** `GET /ratings/fpi`

**Returns:** `year, team, conference, fpi, efficiencies_overall, efficiencies_offense, efficiencies_defense, efficiencies_special_teams, resume_ranks_strength_of_record, resume_ranks_fpi, resume_ranks_average_win_probability, resume_ranks_strength_of_schedule, resume_ranks_remaining_strength_of_schedule, resume_ranks_game_control`

```python
# utils/cfbd_client.py

def get_fpi_ratings(year: int) -> list:
    """Get ESPN FPI ratings for a given year."""
    api = cfbd.RatingsApi(_client())
    try:
        return api.get_fpi_ratings(year=year)
    except Exception as e:
        logger.error(f"CFBD get_fpi_ratings error: {e}")
        return []
```

```python
# Feature engineering
def _add_fpi_features(df: pd.DataFrame, fpi_df: pd.DataFrame) -> pd.DataFrame:
    """Add FPI-based power rating features."""
    fpi = fpi_df[["team", "fpi", "efficiencies_overall",
                   "efficiencies_offense", "efficiencies_defense",
                   "efficiencies_special_teams"]].copy()

    # Merge for home team
    home = fpi.rename(columns={c: f"home_{c}" for c in fpi.columns if c != "team"})
    home = home.rename(columns={"team": "home_team"})
    df = df.merge(home, on="home_team", how="left")

    # Merge for away team
    away = fpi.rename(columns={c: f"away_{c}" for c in fpi.columns if c != "team"})
    away = away.rename(columns={"team": "away_team"})
    df = df.merge(away, on="away_team", how="left")

    # Diffs
    df["fpi_diff"] = df["home_fpi"] - df["away_fpi"]
    df["fpi_off_diff"] = df["home_efficiencies_offense"] - df["away_efficiencies_offense"]
    df["fpi_def_diff"] = df["home_efficiencies_defense"] - df["away_efficiencies_defense"]
    df["fpi_st_diff"] = df["home_efficiencies_special_teams"] - df["away_efficiencies_special_teams"]
    return df
```

---

### 1d. SRS (Simple Rating System) Ratings ✅

cfbfastR wraps `cfbd_ratings_srs()` — a margin-of-victory + strength-of-schedule adjusted rating. Complementary to SP+ and Elo.

**CFBD API endpoint:** `GET /ratings/srs`

**Returns:** `year, team, conference, division, rating, ranking`

```python
# utils/cfbd_client.py

def get_srs_ratings(year: int) -> list:
    """Get Simple Rating System (SRS) ratings for a given year."""
    api = cfbd.RatingsApi(_client())
    try:
        return api.get_srs_ratings(year=year)
    except Exception as e:
        logger.error(f"CFBD get_srs_ratings error: {e}")
        return []
```

```python
# Feature engineering
def _add_srs_features(df: pd.DataFrame, srs_df: pd.DataFrame) -> pd.DataFrame:
    """Add SRS rating diff as a feature."""
    srs = srs_df[["team", "rating"]].rename(columns={"rating": "srs_rating"})

    home_srs = srs.rename(columns={"team": "home_team", "srs_rating": "home_srs"})
    df = df.merge(home_srs, on="home_team", how="left")

    away_srs = srs.rename(columns={"team": "away_team", "srs_rating": "away_srs"})
    df = df.merge(away_srs, on="away_team", how="left")

    df["srs_diff"] = df["home_srs"].fillna(0) - df["away_srs"].fillna(0)
    return df
```

---

### 1e. Pre-Game Win Probabilities (CFBD Model) ✅

cfbfastR wraps `cfbd_metrics_wp_pregame()` which returns CFBD's own pre-game home/away win probabilities and the spread. This can be used as a **consensus model input** or benchmark.

**CFBD API endpoint:** `GET /metrics/wp/pregame`

**Returns:** `season, season_type, week, game_id, home_team, away_team, spread, home_win_prob, away_win_prob`

```python
# utils/cfbd_client.py

def get_pregame_win_prob(year: int, week: int | None = None) -> list:
    """Get CFBD's pre-game win probabilities."""
    api = cfbd.MetricsApi(_client())
    try:
        kwargs = {"year": year}
        if week is not None:
            kwargs["week"] = week
        return api.get_pregame_win_probabilities(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_pregame_win_prob error: {e}")
        return []
```

```python
# Feature: use CFBD's pregame WP as an ensemble input
# In feature_engine.py, after merging:
# df["cfbd_pregame_wp_diff"] = df["cfbd_home_win_prob"] - 0.5
# This gives the model access to CFBD's own prediction as a feature
```

---

### 1f. PPA / WEPA Metrics (Predicted Points Added, Opponent-Adjusted) ✅

cfbfastR exposes several PPA endpoints not currently used:

| Endpoint | Description |
|----------|-------------|
| `cfbd_metrics_ppa_teams()` | Season team PPA (off/def by down, cumulative) |
| `cfbd_metrics_ppa_games()` | Per-game team PPA |
| `cfbd_metrics_wepa_team_season()` | **Opponent-adjusted** team PPA (WEPA) |
| `cfbd_metrics_wepa_players_passing()` | Opponent-adjusted player passing PPA |
| `cfbd_metrics_wepa_players_rushing()` | Opponent-adjusted player rushing PPA |
| `cfbd_metrics_wepa_players_kicking()` | Kicker PAAR (Points Added Above Replacement) |

**Key insight:** WEPA is opponent-adjusted, making it superior to raw EPA for modeling. The current project uses raw `off_epa_diff` but not opponent-adjusted versions.

**CFBD API endpoint:** `GET /ppa/teams` and `GET /ppa/teams/adjusted`

```python
# utils/cfbd_client.py

def get_ppa_teams(year: int, team: str | None = None,
                  conference: str | None = None,
                  excl_garbage_time: bool = True) -> list:
    """Get team PPA (predicted points added) averages for a season."""
    api = cfbd.MetricsApi(_client())
    try:
        kwargs = {"year": year, "excl_garbage_time": excl_garbage_time}
        if team:
            kwargs["team"] = team
        if conference:
            kwargs["conference"] = conference
        return api.get_team_ppa(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_ppa_teams error: {e}")
        return []


def get_wepa_team_season(year: int, team: str | None = None) -> list:
    """Get opponent-adjusted (WEPA) team season PPA stats."""
    api = cfbd.MetricsApi(_client())
    try:
        kwargs = {"year": year}
        if team:
            kwargs["team"] = team
        return api.get_adjusted_team_season_stats(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_wepa_team_season error: {e}")
        return []
```

```python
# Feature engineering — PPA by down (reveals situational efficiency)
def _add_ppa_features(df: pd.DataFrame, ppa_df: pd.DataFrame) -> pd.DataFrame:
    """Add PPA-based features broken down by down."""
    cols = ["team", "off_overall", "off_passing", "off_rushing",
            "off_first_down", "off_second_down", "off_third_down",
            "def_overall", "def_passing", "def_rushing",
            "def_first_down", "def_second_down", "def_third_down"]
    ppa = ppa_df[cols].copy()

    for col in cols[1:]:
        ppa[col] = pd.to_numeric(ppa[col], errors="coerce")

    # Home
    home = ppa.rename(columns={c: f"home_ppa_{c}" if c != "team" else "home_team" for c in cols})
    df = df.merge(home, on="home_team", how="left")

    # Away
    away = ppa.rename(columns={c: f"away_ppa_{c}" if c != "team" else "away_team" for c in cols})
    df = df.merge(away, on="away_team", how="left")

    # Key diffs
    df["ppa_off_diff"] = df["home_ppa_off_overall"] - df["away_ppa_off_overall"]
    df["ppa_def_diff"] = df["home_ppa_def_overall"] - df["away_ppa_def_overall"]
    df["ppa_third_down_off_diff"] = (
        df["home_ppa_off_third_down"] - df["away_ppa_off_third_down"]
    )
    df["ppa_third_down_def_diff"] = (
        df["home_ppa_def_third_down"] - df["away_ppa_def_third_down"]
    )
    return df
```

---

### 1g. Player Returning Production ✅

cfbfastR wraps `cfbd_player_returning()` — the percentage of a team's production (passing, rushing, receiving, PPA) returning from the prior year. **Extremely valuable for preseason / early-season predictions.**

**CFBD API endpoint:** `GET /player/returning`

**Returns:** `season, team, conference, total_ppa, total_passing_ppa, total_receiving_ppa, total_rushing_ppa, percent_ppa, percent_passing_ppa, percent_receiving_ppa, percent_rushing_ppa, usage, passing_usage, receiving_usage, rushing_usage`

```python
# utils/cfbd_client.py

def get_returning_production(year: int) -> list:
    """Get player returning production metrics for a given year."""
    api = cfbd.PlayersApi(_client())
    try:
        return api.get_returning_production(year=year)
    except Exception as e:
        logger.error(f"CFBD get_returning_production error: {e}")
        return []
```

```python
# Feature engineering — returning production
def _add_returning_production(df: pd.DataFrame, ret_df: pd.DataFrame) -> pd.DataFrame:
    """Add returning production features (crucial for early-season games)."""
    ret = ret_df[["team", "percent_ppa", "percent_passing_ppa",
                   "percent_receiving_ppa", "percent_rushing_ppa"]].copy()
    for col in ret.columns[1:]:
        ret[col] = pd.to_numeric(ret[col], errors="coerce")

    # Home
    home = ret.rename(columns={"team": "home_team",
                                "percent_ppa": "home_ret_ppa_pct",
                                "percent_passing_ppa": "home_ret_pass_pct",
                                "percent_receiving_ppa": "home_ret_recv_pct",
                                "percent_rushing_ppa": "home_ret_rush_pct"})
    df = df.merge(home, on="home_team", how="left")

    # Away
    away = ret.rename(columns={"team": "away_team",
                                "percent_ppa": "away_ret_ppa_pct",
                                "percent_passing_ppa": "away_ret_pass_pct",
                                "percent_receiving_ppa": "away_ret_recv_pct",
                                "percent_rushing_ppa": "away_ret_rush_pct"})
    df = df.merge(away, on="away_team", how="left")

    # Key diff: overall returning PPA %
    df["returning_ppa_diff"] = (
        df["home_ret_ppa_pct"].fillna(0.5) - df["away_ret_ppa_pct"].fillna(0.5)
    )
    return df
```

---

### 1h. Transfer Portal Data ✅

cfbfastR wraps `cfbd_recruiting_transfer_portal()` — players entering the portal with ratings, positions, origin/destination.

**CFBD API endpoint:** `GET /player/portal`

```python
# utils/cfbd_client.py

def get_transfer_portal(year: int) -> list:
    """Get transfer portal data for a given year."""
    api = cfbd.PlayersApi(_client())
    try:
        return api.get_transfer_portal(year=year)
    except Exception as e:
        logger.error(f"CFBD get_transfer_portal error: {e}")
        return []
```

```python
# Aggregate portal data into team-level talent gains/losses
def _aggregate_portal_impact(portal_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize net transfer portal impact per team."""
    portal = portal_df.copy()
    portal["rating"] = pd.to_numeric(portal.get("rating", 0), errors="coerce").fillna(0)

    # Gains: players transferring TO a team
    gains = (portal.groupby("destination")["rating"]
             .agg(portal_gains_sum="sum", portal_gains_count="count")
             .reset_index()
             .rename(columns={"destination": "team"}))

    # Losses: players transferring FROM a team
    losses = (portal.groupby("origin")["rating"]
              .agg(portal_losses_sum="sum", portal_losses_count="count")
              .reset_index()
              .rename(columns={"origin": "team"}))

    merged = gains.merge(losses, on="team", how="outer").fillna(0)
    merged["portal_net_rating"] = merged["portal_gains_sum"] - merged["portal_losses_sum"]
    merged["portal_net_count"] = merged["portal_gains_count"] - merged["portal_losses_count"]
    return merged
```

---

### 1i. Game Media / TV Coverage ✅

cfbfastR wraps `cfbd_game_media()` — TV/radio/web broadcast info. TV games tend to have higher visibility and potentially different dynamics (prime-time letdown, etc.).

**CFBD API endpoint:** `GET /games/media`

```python
# utils/cfbd_client.py

def get_game_media(year: int, week: int | None = None) -> list:
    """Get game media/broadcast information."""
    api = cfbd.GamesApi(_client())
    try:
        kwargs = {"year": year}
        if week is not None:
            kwargs["week"] = week
        return api.get_game_media(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_game_media error: {e}")
        return []
```

```python
# Feature: prime-time indicator (impacts betting / performance)
def _add_media_features(df: pd.DataFrame, media_df: pd.DataFrame) -> pd.DataFrame:
    """Add TV/prime-time features."""
    mdf = media_df.copy()
    # Prime-time networks
    prime_networks = {"ESPN", "ABC", "FOX", "CBS", "NBC", "ESPN2"}
    mdf["is_primetime"] = mdf["tv"].apply(
        lambda x: int(any(net in str(x) for net in prime_networks)) if x else 0
    )
    return df.merge(mdf[["game_id", "is_primetime"]], on="game_id", how="left")
```

---

### 1j. Play-by-Play Data ✅ (API stub)

**The single biggest feature gap.** cfbfastR's core value proposition is play-by-play data with EPA and WPA columns. The current project uses only game-level and season-level aggregates.

**CFBD API endpoint:** `GET /plays`

```python
# utils/cfbd_client.py

def get_plays(year: int, week: int, season_type: str = "regular",
              team: str | None = None) -> list:
    """Get play-by-play data for a specific year/week."""
    api = cfbd.PlaysApi(_client())
    try:
        kwargs = {
            "year": year,
            "week": week,
            "season_type": cfbd.SeasonType(season_type),
        }
        if team:
            kwargs["team"] = team
        return api.get_plays(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_plays error: {e}")
        return []


def get_play_stats(year: int, week: int, season_type: str = "regular") -> list:
    """Get player-level play stats."""
    api = cfbd.PlaysApi(_client())
    try:
        return api.get_play_stat_types()
    except Exception as e:
        logger.error(f"CFBD get_play_stats error: {e}")
        return []
```

```python
# Derive custom EPA features from PBP (see section 3a for full model)
def _compute_pbp_features(pbp_df: pd.DataFrame, game_id: int) -> dict:
    """Compute game-level features from play-by-play data."""
    game_plays = pbp_df[pbp_df["game_id"] == game_id].copy()

    home = game_plays[game_plays["home_away"] == "home"]
    away = game_plays[game_plays["home_away"] == "away"]

    return {
        "game_id": game_id,
        # Success rate: % of plays with positive EPA
        "home_success_rate": (home["epa"] > 0).mean() if len(home) > 0 else 0.5,
        "away_success_rate": (away["epa"] > 0).mean() if len(away) > 0 else 0.5,
        # Explosive play rate: EPA > 2.0
        "home_explosive_rate": (home["epa"] > 2.0).mean() if len(home) > 0 else 0,
        "away_explosive_rate": (away["epa"] > 2.0).mean() if len(away) > 0 else 0,
        # Stuff rate: rush plays with <= 0 yards
        "home_stuff_rate": (
            (home[(home["play_type"] == "Rush")]["yards_gained"] <= 0).mean()
            if len(home[home["play_type"] == "Rush"]) > 0 else 0
        ),
        # Red zone plays
        "home_rz_plays": len(home[home["yards_to_goal"] <= 20]),
        "away_rz_plays": len(away[away["yards_to_goal"] <= 20]),
    }
```

---

### 1k. Drive-Level Data ✅ (API stub)

cfbfastR wraps `cfbd_drives()` — drive start/end yard line, result, time of possession, number of plays per drive.

**CFBD API endpoint:** `GET /drives`

```python
# utils/cfbd_client.py

def get_drives(year: int, week: int, season_type: str = "regular",
               team: str | None = None) -> list:
    """Get drive-by-drive data."""
    api = cfbd.DrivesApi(_client())
    try:
        kwargs = {
            "year": year,
            "week": week,
            "season_type": cfbd.SeasonType(season_type),
        }
        if team:
            kwargs["team"] = team
        return api.get_drives(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_drives error: {e}")
        return []
```

```python
# Feature engineering from drives
def _compute_drive_features(drives_df: pd.DataFrame, game_id: int) -> dict:
    """Derive drive-level efficiency metrics."""
    game_drives = drives_df[drives_df["game_id"] == game_id]

    scoring_drives = game_drives[game_drives["scoring"] == True]
    results = game_drives["drive_result"].value_counts()

    return {
        "game_id": game_id,
        "scoring_drive_pct": len(scoring_drives) / max(len(game_drives), 1),
        "avg_drive_plays": game_drives["plays"].mean(),
        "avg_drive_yards": game_drives["yards"].mean(),
        "three_and_out_pct": results.get("Punt", 0) / max(len(game_drives), 1),
        "turnover_drive_pct": (
            results.get("Fumble", 0) + results.get("Interception", 0)
        ) / max(len(game_drives), 1),
    }
```

---

### 1l. Player Usage Metrics ✅ (API stub)

cfbfastR wraps `cfbd_player_usage()` — per-player usage rates, PPA contributions.

**CFBD API endpoint:** `GET /player/usage`

```python
# utils/cfbd_client.py

def get_player_usage(year: int, team: str | None = None) -> list:
    """Get player usage metrics (snap/play share, PPA contribution)."""
    api = cfbd.PlayersApi(_client())
    try:
        kwargs = {"year": year}
        if team:
            kwargs["team"] = team
        return api.get_player_usage(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_player_usage error: {e}")
        return []
```

---

### 1m. Game Player Stats ✅ (API stub)

cfbfastR wraps `cfbd_game_player_stats()` — player-level box score stats per game (passing/rushing/receiving/defense).

**CFBD API endpoint:** `GET /games/players`

```python
# utils/cfbd_client.py

def get_game_player_stats(year: int, week: int | None = None,
                          season_type: str = "regular",
                          team: str | None = None) -> list:
    """Get player statistics by game."""
    api = cfbd.GamesApi(_client())
    try:
        kwargs = {
            "year": year,
            "season_type": cfbd.SeasonType(season_type),
        }
        if week:
            kwargs["week"] = week
        if team:
            kwargs["team"] = team
        return api.get_player_game_stats(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_game_player_stats error: {e}")
        return []
```

---

### 1n. Coach History ✅

cfbfastR wraps `cfbd_coaches()` — historical coaching records.

**CFBD API endpoint:** `GET /coaches`

```python
# utils/cfbd_client.py

def get_coaches(year: int | None = None, team: str | None = None) -> list:
    """Get coaching history and records."""
    api = cfbd.CoachesApi(_client())
    try:
        kwargs = {}
        if year:
            kwargs["year"] = year
        if team:
            kwargs["team"] = team
        return api.get_coaches(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_coaches error: {e}")
        return []
```

```python
# Feature: first-year coach penalty
def _add_coach_features(df: pd.DataFrame, coaches_df: pd.DataFrame) -> pd.DataFrame:
    """Flag first-year coaches (historically perform worse)."""
    coaches = coaches_df.copy()
    # seasons is a list; first-year = only 1 season for current year
    coaches["tenure_years"] = coaches["seasons"].apply(
        lambda x: len(x) if isinstance(x, list) else 1
    )
    coaches["first_year_coach"] = (coaches["tenure_years"] == 1).astype(int)

    coach_cols = coaches[["team", "first_year_coach", "tenure_years"]].copy()

    # Merge for home
    home_c = coach_cols.rename(columns={"team": "home_team",
                                         "first_year_coach": "home_first_yr_coach",
                                         "tenure_years": "home_coach_tenure"})
    df = df.merge(home_c, on="home_team", how="left")

    # Merge for away
    away_c = coach_cols.rename(columns={"team": "away_team",
                                         "first_year_coach": "away_first_yr_coach",
                                         "tenure_years": "away_coach_tenure"})
    df = df.merge(away_c, on="away_team", how="left")

    df["coach_tenure_diff"] = (
        df["home_coach_tenure"].fillna(3) - df["away_coach_tenure"].fillna(3)
    )
    return df
```

---

### 1o. Head-to-Head Matchup Records (API-native) ✅

cfbfastR wraps `cfbd_team_matchup()` and `cfbd_team_matchup_records()`. Currently the Historical Analysis page does H2H manually via game filtering. The API can return this natively.

**CFBD API endpoint:** `GET /teams/matchup`

```python
# utils/cfbd_client.py

def get_team_matchup(team1: str, team2: str,
                     min_year: int | None = None,
                     max_year: int | None = None) -> dict:
    """Get head-to-head matchup history between two teams."""
    api = cfbd.TeamsApi(_client())
    try:
        kwargs = {"team1": team1, "team2": team2}
        if min_year:
            kwargs["min_year"] = min_year
        if max_year:
            kwargs["max_year"] = max_year
        return api.get_team_matchup(**kwargs)
    except Exception as e:
        logger.error(f"CFBD get_team_matchup error: {e}")
        return {}
```

---

## 2. New Features for Modeling

Summary of new features that can be derived from the data sources above, grouped by impact:

### High Impact (add to WIN_FEATURES and SPREAD_FEATURES)

| Feature | Source | Rationale |
|---------|--------|-----------|
| `fpi_diff` | FPI ratings | ESPN's power index, complementary to SP+ |
| `srs_diff` | SRS ratings | Margin + SOS adjusted, proven predictive |
| `returning_ppa_diff` | Returning production | Critical for early-season when rolling stats are sparse |
| `ppa_third_down_off_diff` | PPA by down | Third-down efficiency is highly predictive of game outcomes |
| `wepa_off_diff` / `wepa_def_diff` | WEPA | Opponent-adjusted EPA is superior to raw EPA |
| `cfbd_pregame_wp_diff` | Pre-game WP | Consensus model stacking |

### Medium Impact (add to TOTAL_FEATURES and/or WIN_FEATURES)

| Feature | Source | Rationale |
|---------|--------|-----------|
| `is_dome` | Venues/Weather | Dome games have higher totals, different dynamics |
| `temperature` | Weather | Cold weather depresses scoring |
| `wind_speed` | Weather | High wind kills passing game |
| `adverse_weather` | Weather | Composite bad-weather flag |
| `high_altitude` | Venues | Thin air affects kicking, big-play rate |
| `portal_net_rating` | Transfer portal | Talent turnover indicator |
| `coach_tenure_diff` | Coaches | First-year coaches underperform |
| `is_primetime` | Media | National TV games have different ATS trends |

### Lower Impact / Situational

| Feature | Source | Rationale |
|---------|--------|-----------|
| `scoring_drive_pct` | Drives | Drive efficiency proxy |
| `three_and_out_pct` | Drives | Defensive dominance indicator |
| `home_explosive_rate` | PBP | Big-play tendency |
| `home_stuff_rate` | PBP | Run-stopping ability |

---

## 3. New Model Ideas ✅

### 3a. Custom Expected Points (EP / EPA) Model ✅

cfbfastR's `create_epa()` builds EPA from a multinomial logistic regression that predicts the next scoring event (7 outcomes: TD, FG, Safety, Opp_TD, Opp_FG, Opp_Safety, No_Score) based on game state. You can replicate this in Python.

**Model variables (pre-play):**
- `TimeSecsRem` — seconds remaining in half
- `down` (1-4, as factor)
- `distance` — yards to first down
- `yards_to_goal` — yards to end zone
- `log_ydstogo` — log(distance)
- `Under_two` — boolean, under 2 minutes
- `Goal_To_Go` — boolean
- `pos_score_diff_start` — score differential from perspective of possessing team

**Scoring weights:** `[No_Score=0, FG=3, Opp_FG=-3, Opp_Safety=-2, Opp_TD=-7, Safety=2, TD=7]`

**EPA = EP_after - EP_before**

```python
# models/epa_model.py — Custom EPA model in Python

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

SCORING_WEIGHTS = {
    "No_Score": 0, "FG": 3, "Opp_FG": -3,
    "Opp_Safety": -2, "Opp_TD": -7, "Safety": 2, "TD": 7
}
WEIGHT_VECTOR = np.array([0, 3, -3, -2, -7, 2, 7])
CLASSES = ["No_Score", "FG", "Opp_FG", "Opp_Safety", "Opp_TD", "Safety", "TD"]

EP_FEATURES = [
    "TimeSecsRem", "down_2", "down_3", "down_4",
    "distance", "yards_to_goal", "log_ydstogo",
    "under_two", "goal_to_go", "pos_score_diff_start"
]


def _label_next_score(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Label each play with the next scoring event type in its half."""
    # For each play, look forward to find the next score
    df = pbp_df.sort_values(["game_id", "id_play"]).copy()
    df["next_score_type"] = "No_Score"

    for game_id in df["game_id"].unique():
        game = df[df["game_id"] == game_id]
        for half in [1, 2]:
            half_plays = game[game["half"] == half].index
            last_score = "No_Score"
            # Walk backward through plays
            for idx in reversed(half_plays):
                play = df.loc[idx]
                if play.get("scoring_play", False):
                    last_score = _classify_score(play)
                df.loc[idx, "next_score_type"] = last_score

    return df


def _classify_score(play) -> str:
    """Classify a scoring play into one of 7 next-score types."""
    pt = play.get("play_type", "")
    if "Touchdown" in pt:
        if play.get("offense_score_play", False):
            return "TD"
        return "Opp_TD"
    if "Field Goal Good" in pt:
        return "FG"
    if "Safety" in pt:
        return "Safety"
    return "No_Score"


def train_ep_model(pbp_df: pd.DataFrame) -> dict:
    """Train a multinomial EP model on play-by-play data."""
    df = _label_next_score(pbp_df)
    df = df[df["down"].isin([1, 2, 3, 4])].copy()
    df["log_ydstogo"] = np.log(df["distance"].clip(lower=1))
    df["under_two"] = (df["TimeSecsRem"] < 120).astype(int)
    df["goal_to_go"] = (df["yards_to_goal"] == df["distance"]).astype(int)

    # One-hot encode down
    down_dummies = pd.get_dummies(df["down"], prefix="down", dtype=int)
    df = pd.concat([df, down_dummies], axis=1)

    X = df[EP_FEATURES].dropna()
    y = df.loc[X.index, "next_score_type"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=1000, C=1.0
    )
    model.fit(X_scaled, y)

    return {"model": model, "scaler": scaler}


def predict_ep(model_dict: dict, play_state: pd.DataFrame) -> np.ndarray:
    """Predict expected points for a set of play states."""
    X = model_dict["scaler"].transform(play_state[EP_FEATURES])
    probs = model_dict["model"].predict_proba(X)
    # Map class order to weight vector
    class_order = model_dict["model"].classes_
    weights = np.array([SCORING_WEIGHTS[c] for c in class_order])
    return probs @ weights  # Expected points


def compute_epa(model_dict: dict, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute EPA for each play: EP_after - EP_before."""
    df = pbp_df.copy()
    df["ep_before"] = predict_ep(model_dict, df)  # pre-play state
    # Build post-play state (next down/distance/yardline)
    df["ep_after"] = predict_ep(model_dict, _build_after_state(df))
    df["epa"] = df["ep_after"] - df["ep_before"]
    return df
```

---

### 3b. Win Probability Added (WPA) Model ✅

cfbfastR's `create_wpa_naive()` uses a GAM/logistic model predicting win probability from game state. Key features:

**WP Model features:**
- `ExpScoreDiff` = `pos_score_diff_start + ep_before` (expected score differential)
- `ExpScoreDiff_Time_Ratio` = `ExpScoreDiff / (adj_TimeSecsRem + 1)`
- `half` (1 or 2)
- `Under_two` (under 2 min warning)
- `pos_team_timeouts_rem_before`
- `def_pos_team_timeouts_rem_before`
- `pos_score_diff_start`

**WPA = wp_after - wp_before** (from possessing team's perspective)

```python
# models/wpa_model.py — Win Probability model

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

WP_FEATURES = [
    "ExpScoreDiff",
    "ExpScoreDiff_Time_Ratio",
    "half",
    "under_two",
    "pos_team_timeouts_rem",
    "def_pos_team_timeouts_rem",
    "pos_score_diff_start",
]


def _prepare_wp_features(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare WP model features from EPA-enriched PBP data."""
    df = pbp_df.copy()
    df["adj_TimeSecsRem"] = np.where(
        df["half"] == 1,
        1800 + df["TimeSecsRem"],
        df["TimeSecsRem"]
    )
    df["ExpScoreDiff"] = df["pos_score_diff_start"] + df["ep_before"]
    df["ExpScoreDiff_Time_Ratio"] = df["ExpScoreDiff"] / (df["adj_TimeSecsRem"] + 1)
    df["under_two"] = (df["TimeSecsRem"] < 120).astype(int)
    return df


def train_wp_model(pbp_df: pd.DataFrame) -> dict:
    """Train in-game win probability model."""
    df = _prepare_wp_features(pbp_df)

    # Label: did the possessing team's side win?
    df["pos_team_won"] = (df["pos_team"] == df["winner"]).astype(int)

    X = df[WP_FEATURES].dropna()
    y = df.loc[X.index, "pos_team_won"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_scaled, y)

    return {"model": model, "scaler": scaler}


def predict_wp(model_dict: dict, state_df: pd.DataFrame) -> np.ndarray:
    """Predict win probability for possessing team."""
    X = model_dict["scaler"].transform(state_df[WP_FEATURES])
    return model_dict["model"].predict_proba(X)[:, 1]


def compute_wpa(model_dict: dict, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute WPA for each play."""
    df = _prepare_wp_features(pbp_df)
    df["wp_before"] = predict_wp(model_dict, df)
    # Shift to get wp of next play as wp_after
    df["wp_after"] = df.groupby("game_id")["wp_before"].shift(-1)
    # Account for change of possession
    df["wpa"] = np.where(
        df["pos_team"] == df["pos_team"].shift(-1),
        df["wp_after"] - df["wp_before"],
        (1 - df["wp_after"]) - df["wp_before"]
    )
    # Home perspective
    df["home_wp"] = np.where(
        df["pos_team"] == df["home"],
        df["wp_before"],
        1 - df["wp_before"]
    )
    return df
```

---

### 3c. Field Goal Expected Points Model ✅

cfbfastR uses a GAM (`mgcv::bam`) to model FG make probability based on `yards_to_goal`. This adjusts EP calculations for FG attempts — weighting the EP of a made FG vs. the EP of a miss (opponent gets ball at LOS + 8 yards).

```python
# models/fg_model.py — Field Goal probability model

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def train_fg_model(pbp_df: pd.DataFrame) -> IsotonicRegression:
    """Train FG make probability model based on distance."""
    fg_plays = pbp_df[pbp_df["play_type"].str.contains("Field Goal", na=False)].copy()
    fg_plays["fg_distance"] = fg_plays["yards_to_goal"] + 17  # snap + hold
    fg_plays["fg_made"] = fg_plays["play_type"].str.contains("Good").astype(int)

    # Isotonic regression: monotonically decreasing make% with distance
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(fg_plays["fg_distance"].values, fg_plays["fg_made"].values)
    return model


def predict_fg_prob(model: IsotonicRegression, distance: float) -> float:
    """Predict probability of making a field goal from given distance."""
    return float(model.predict(np.array([distance]))[0])
```

---

## 4. New Pages / Visualizations ✅

Ideas inspired by cfbfastR's vignettes and outputs:

### 4a. Win Probability Chart Page ✅

Display an interactive in-game win probability chart for any historical game, similar to cfbfastR's `cfbd_metrics_wp()`.

```python
# pages/7_Win_Probability.py

import streamlit as st
import plotly.graph_objects as go
from utils.cfbd_client import get_win_probability_chart

# Add to cfbd_client.py:
# def get_win_probability_chart(game_id: int) -> list:
#     api = cfbd.MetricsApi(_client())
#     return api.get_win_probability_data(game_id=game_id)

st.set_page_config(page_title="Win Probability", layout="wide")
st.title("Win Probability Chart")

game_id = st.number_input("Game ID", value=401403874)
wp_data = get_win_probability_chart(game_id)

if wp_data:
    plays = [{"play_number": i, "home_wp": p.home_win_prob} for i, p in enumerate(wp_data)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p["play_number"] for p in plays],
        y=[p["home_wp"] for p in plays],
        mode="lines",
        fill="tozeroy",
        name="Home Win Prob"
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        yaxis_title="Home Win Probability",
        xaxis_title="Play Number",
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig, use_container_width=True)
```

### 4b. EPA Scatter / Efficiency Dashboard

Plot offense EPA vs defense EPA with teams as points — top-right quadrant = elite teams.

```python
# In Team Explorer or new page
import plotly.express as px

def plot_epa_scatter(advanced_df: pd.DataFrame, season: int):
    """Plot off EPA vs def EPA scatter (team efficiency quadrant)."""
    df = advanced_df[advanced_df["season"] == season].copy()
    fig = px.scatter(
        df, x="off_epa_per_play", y="def_epa_per_play",
        text="team", color="conference",
        title=f"{season} Team Efficiency: Offense EPA vs Defense EPA",
        labels={
            "off_epa_per_play": "Offensive EPA/Play (higher = better)",
            "def_epa_per_play": "Defensive EPA/Play (lower = better)"
        }
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    # Annotate quadrants
    fig.add_annotation(x=0.15, y=-0.15, text="ELITE", showarrow=False,
                       font=dict(size=16, color="green"))
    fig.add_annotation(x=-0.15, y=0.15, text="REBUILDING", showarrow=False,
                       font=dict(size=16, color="red"))
    return fig
```

### 4c. Returning Production Preseason Page ✅

Show which teams return the most production — critical for early-season predictions.

```python
# pages/8_Preseason_Outlook.py (snippet)
def show_returning_production(ret_df: pd.DataFrame, season: int):
    """Display returning production leaderboard."""
    df = ret_df[ret_df["season"] == season].sort_values("percent_ppa", ascending=False)
    st.dataframe(
        df[["team", "conference", "percent_ppa", "percent_passing_ppa",
            "percent_rushing_ppa", "percent_receiving_ppa"]].head(25),
        column_config={
            "percent_ppa": st.column_config.ProgressColumn("Total PPA Returning %",
                                                            min_value=0, max_value=1,
                                                            format="%.0f%%"),
        },
        use_container_width=True
    )
```

---

## 5. Implementation Priority Matrix

> **Implementation complete** — all items below have been implemented. Model impact vs baseline:
> - Spread RMSE: 13.80 → **13.73** (−0.07, improved)
> - Win log_loss: 0.489 → **0.488** (−0.001, improved)
> - Win Brier: 0.1602 → **0.1608** (flat; weather features unavailable without Patreon Tier 1)
> - ATS record: 64.9% → **65.1%** (+0.2%)
> - Note: Weather data requires CFBD Patreon Tier 1+; features `is_dome`, `temperature`, `wind_speed`, `adverse_weather`, `high_wind` will activate automatically once available.

| Priority | Item | Effort | Impact | Status |
|----------|------|--------|--------|--------|
| **P0** | Weather data (`get_game_weather`) | Low | High | ✅ API stub — requires Patreon Tier 1 |
| **P0** | FPI ratings (`get_fpi_ratings`) | Low | High | ✅ 664 rows fetched, `fpi_diff` in WIN_FEATURES |
| **P0** | SRS ratings (`get_srs_ratings`) | Low | High | ✅ 1,183 rows fetched, `srs_diff` in WIN_FEATURES |
| **P0** | Returning production (`get_returning_production`) | Low | High | ✅ 656 rows, `returning_ppa_diff` in WIN_FEATURES |
| **P1** | Pre-game WP as feature | Low | Medium | ✅ 4,513 rows, `cfbd_pregame_wp_diff` in WIN_FEATURES |
| **P1** | PPA by down (`get_ppa_teams`) | Low | Medium | ✅ 664 rows, 4 PPA features in WIN_FEATURES |
| **P1** | WEPA opponent-adjusted metrics | Low | High | ✅ API stub — not available in SDK |
| **P1** | Coach tenure feature | Low | Medium | ✅ 664 rows, `coach_tenure_diff` in WIN_FEATURES |
| **P2** | Transfer portal aggregation | Medium | Medium | ✅ 14,422 rows, portal net features added |
| **P2** | Venue features (elevation, dome) | Low | Low-Med | ✅ 840 venues, `high_altitude`, `is_dome` in TOTAL_FEATURES |
| **P2** | Game media / prime-time flag | Low | Low | ✅ 7,808 rows, `is_primetime` in TOTAL_FEATURES |
| **P3** | Play-by-play ingestion | High | High | ✅ `get_plays()` API stub in cfbd_client |
| **P3** | Drive-level features | Medium | Medium | ✅ `get_drives()` API stub in cfbd_client |
| **P3** | Custom EPA model | High | High | ✅ `models/epa_model.py` created |
| **P3** | WPA model / WP chart page | High | Medium | ✅ `models/wpa_model.py` + `pages/7_Win_Probability.py` |
| **P3** | H2H matchup via API | Low | Low | ✅ `get_team_matchup()` in cfbd_client |
| **P4** | Player usage metrics | Medium | Low | ✅ `get_player_usage()` API stub in cfbd_client |
| **P4** | FG EP model | Medium | Low | ✅ `models/fg_model.py` created |

---

## Quick Start: Adding P0 Items

To rapidly integrate the four P0 items, add these fetches to the data collection loop in `utils/storage.py`:

```python
# In the main data collection function, add alongside existing API calls:

for year in range(START_YEAR, END_YEAR + 1):
    # ... existing calls ...

    # NEW: Weather
    weather = cfbd_client.get_game_weather(year)
    save_raw_json([vars(w) if hasattr(w, '__dict__') else w for w in weather],
                  f"weather_{year}.json")

    # NEW: FPI ratings
    fpi = cfbd_client.get_fpi_ratings(year)
    save_raw_json([vars(f) if hasattr(f, '__dict__') else f for f in fpi],
                  f"fpi_ratings_{year}.json")

    # NEW: SRS ratings
    srs = cfbd_client.get_srs_ratings(year)
    save_raw_json([vars(s) if hasattr(s, '__dict__') else s for s in srs],
                  f"srs_ratings_{year}.json")

    # NEW: Returning production
    ret = cfbd_client.get_returning_production(year)
    save_raw_json([vars(r) if hasattr(r, '__dict__') else r for r in ret],
                  f"returning_production_{year}.json")
```

Then update `feature_engine.py`'s `build_feature_matrix()` to load and merge these new tables, using the feature engineering functions defined above.

Add the new features to the model feature lists in `utils/models.py`:

```python
# utils/models.py — extend feature lists

# Add to WIN_FEATURES:
EXTRA_WIN_FEATURES = [
    "fpi_diff",
    "srs_diff",
    "returning_ppa_diff",
    "wepa_off_diff",       # if WEPA data fetched
    "coach_tenure_diff",   # if coach data fetched
]

# Add to TOTAL_FEATURES:
EXTRA_TOTAL_FEATURES = [
    "is_dome",
    "temperature",
    "wind_speed",
    "adverse_weather",
]
```
