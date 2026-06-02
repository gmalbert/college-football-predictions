# Tailgate Edge — Architecture

## Overview
College football (NCAAF) predictions and betting analytics platform. Uses Elo ratings, team efficiency metrics, and historical CFBD data to predict game outcomes and surface betting value.

## Data Flow
```
CFBD API (college-football-data)    ESPN API    The Odds API
        ↓                               ↓               ↓
utils/cfbd_client.py          utils/espn_client.py   odds fetch
        ↓                               ↓               ↓
            data_files/*.parquet (historical games)
                        ↓
            utils/feature_engine.py
                        ↓
        Elo ratings + XGBoost / Logistic Regression
                        ↓
            data_files/best_bets_today.json
                        ↓
            Streamlit pages → predictions.py (entry)
```

## ML Models
- **Elo Engine** (`utils/elo.py`): 1500 default, home +65, 1/3 regression to mean each offseason
- **Logistic Regression**: win probability from Elo + efficiency differentials
- **XGBoost**: trained on 13+ features for spread/total predictions
- Edge = model probability − market implied probability

## API Integrations
| Source | Purpose | Key |
|--------|---------|-----|
| CFBD API | Historical games, schedules, team stats | `CFBD_API_KEY` |
| ESPN API | Live scores, game details | None (public) |
| The Odds API | DraftKings lines (moneyline, spread, total) | `ODDS_API_KEY` |

## Key Components
- `predictions.py` — entry, `st.set_page_config`, sidebar, `st.navigation`
- `utils/cfbd_client.py` — all CFBD data fetching
- `utils/elo.py` — Elo rating engine with season regression
- `utils/feature_engine.py` — feature engineering pipeline
- `utils/models.py` — model training, evaluation, prediction
- `utils/betting.py` — `expected_value()`, `kelly_criterion()`, tier classification
- `utils/storage.py` — Parquet I/O helpers
- `utils/config.py` — constants (thresholds, endpoints)
- `scripts/export_best_bets.py` — writes `data_files/best_bets_today.json`

## Conference Structure
- Power 4: ACC, Big Ten, Big 12, SEC
- Group of 5: AAC, C-USA, MAC, Mountain West, Sun Belt
- Independents: Notre Dame, Army, Navy, UMass

## Storage
- Parquet files in `data_files/` for historical game data
- `data_files/best_bets_today.json` — Sports Picks Grid aggregator feed
