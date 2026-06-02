# College Football Oracle — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: Open-Meteo Weather Integration for Outdoor Stadiums

**Why:** Weather is a top-5 predictor for college football totals. Cold, windy games at outdoor stadiums in November significantly reduce scoring. Open-Meteo is free and requires no API key — it is the easiest high-value feature to add.

**How:**
1. Add `scripts/fetch_weather.py` using Open-Meteo's forecast endpoint: `https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,wind_speed_10m,precipitation`
2. Maintain `data_files/stadium_coords.csv` mapping team → lat/lon (manually maintained, ~135 FBS teams)
3. Compute per-game: `temperature_f`, `wind_speed_mph`, `is_cold` (≤35°F), `is_windy` (≥15mph), `is_dome`
4. Add these to the model feature set in `prepare_model_data.py` for upcoming fixture predictions

**Complexity:** Medium

---

## Feature 2: Play-by-Play EPA Features from CFBD

**Why:** EPA (Expected Points Added) per play is the most reliable predictor of future offensive/defensive performance. `docs/cfbfastr_integration_ideas.md` documents this as a planned feature but it has not been implemented. It would be the single biggest accuracy improvement to the existing feature set.

**How:**
1. Add `scripts/fetch_cfbd_epa.py` using the College Football Data API v5 (`api.collegefootballdata.com/plays`)
2. Compute per-team rolling EPA: `home_off_epa_l5`, `home_def_epa_l5`, `away_off_epa_l5`, `away_def_epa_l5`
3. Add to `prepare_model_data.py` feature engineering alongside existing stats
4. Use `shift(1)` to prevent data leakage (apply EPA from games prior to the current matchup)

**Complexity:** High

---

## Feature 3: Ranked vs Unranked Trap Game Feature

**Why:** Ranked college football teams playing home games against unranked opponents are historically vulnerable to upsets — especially when they have a major rivalry game the following week. This "trap game" dynamic is underpriced by the market.

**How:**
1. Fetch AP Poll and Coaches Poll rankings from the CFBD API (`/rankings` endpoint)
2. Create binary features: `home_ranked`, `away_ranked`, `is_ranked_vs_unranked`
3. Compute `home_next_game_ranked` (1 if the home team's next game is against a ranked opponent — trap game indicator)
4. Add these to the model feature set and validate AUC improvement

**Complexity:** Low

---

## Feature 4: Conference Strength Adjustment (SOS Normalization)

**Why:** SEC offensive stats look very different from MAC offensive stats. When SEC teams play cross-conference matchups (bowl games, non-conference), raw stats systematically misestimate performance. An SOS-adjusted feature would improve bowl game and playoff predictions significantly.

**How:**
1. Fetch strength-of-schedule data from CFBD API (`/teams/matchup` or compute from win%/EPA of opponents faced)
2. Compute `sos_score` per team (average of opponents' EPA)
3. Create SOS-adjusted features: `home_adj_off_epa = home_off_epa / sos_home_defense`, similarly for defense
4. Use adjusted features for bowl game predictions specifically (flag `is_bowl_game` in the feature set)

**Complexity:** Medium

---

## Feature 5: Daily Pipeline Upgrade (ESPN Fixture Fetch)

**Why:** The current GitHub Actions workflow runs weekly. College football games are scheduled days and weeks in advance — switching to a daily run with ESPN's NCAAF scoreboard API would catch late line moves and injury reports earlier in the week.

**How:**
1. Update `.github/workflows/` to trigger daily at 08:00 UTC during the season (September–January)
2. In `scripts/fetch_upcoming_fixtures.py`, add fallback to ESPN NCAAF endpoint: `site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates={week}`
3. Add conditional logic: only retrain the model if new historical results are available (skip if only new upcoming fixtures are added)
4. Add a "Data last updated" timestamp to the Streamlit sidebar

**Complexity:** Low
