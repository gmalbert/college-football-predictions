# Tailgate Edge — GitHub Copilot Instructions

## Project Overview

**App name:** Tailgate Edge
**Purpose:** College football predictions and betting analytics platform. Uses Elo ratings, team efficiency metrics, and historical data to predict NCAAF game outcomes and surface betting value.
**Entry point:** `streamlit run predictions.py`
**Part of:** Betting Oracle suite

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit ≥ 1.36 (multi-page via `st.navigation`) |
| ML | Elo ratings, logistic regression, XGBoost |
| Data | pandas, college-football-data (CFBD) API, ESPN API |
| Odds | The Odds API |
| Config | python-dotenv (`.env` file) |
| Python | 3.9+ |

---

## File Conventions

### Key files
- `predictions.py` — entry point; sets `st.set_page_config` ONCE. Imports `render_sidebar` from `utils/ui_components.py`.
- `utils/cfbd_client.py` — CFBD API client (all historical/schedule data).
- `utils/espn_client.py` — ESPN API client (live scores, game details).
- `utils/elo.py` — Elo rating engine.
- `utils/feature_engine.py` — feature engineering for ML models.
- `utils/models.py` — `load_metrics()`, `models_trained()`, training/prediction logic.
- `utils/storage.py` — `load_parquet()` and parquet I/O helpers.
- `utils/betting.py` — `expected_value()`, `kelly_criterion()`, tier classification.
- `utils/config.py` — constants, thresholds, API endpoints.
- `footer.py` — `add_betting_oracle_footer()` called at page bottom.

### Pages
- `pages/1_Weekly_Predictions.py` — upcoming game predictions + value bets
- `pages/2_Value_Bets.py` — filtered value bet finder
- `pages/3_Team_Explorer.py` — per-team drill-down
- `pages/4_Historical_Analysis.py` — historical model performance
- `pages/5_Model_Performance.py` — accuracy, calibration, ROI metrics
- `pages/6_Settings.py` — user-configurable thresholds
- `pages/7_Win_Probability.py` — head-to-head win probability calculator
- `pages/8_Preseason_Outlook.py` — preseason rankings and outlook

### Data files
- `data_files/logo.png` — app logo
- `data_files/best_bets_today.json` — unified schema for Sports Picks Grid aggregator
- Parquet files in `data_files/` for historical game data

---

## NCAAF Domain Knowledge

### Key bet types
- `moneyline` — outright game winner
- `spread` — point spread
- `total` — over/under total points

### Conference structure
- Power 4: ACC, Big Ten, Big 12, SEC
- Group of 5: AAC, C-USA, MAC, Mountain West, Sun Belt
- Independents: Notre Dame, Army, Navy, UMass

### Elo model
- Default starting Elo: 1500
- Home advantage: ~65 Elo points
- Regression to mean between seasons: revert 1/3 toward 1500

---

## Coding Conventions

### Streamlit patterns
```python
@st.cache_data(ttl=3600)
def load_something() -> pd.DataFrame: ...
```
- `st.set_page_config()` called ONCE in `predictions.py` only
- Individual page files must NOT call `st.set_page_config`
- Use `width='stretch'` for dataframes/charts (not deprecated `use_container_width`)
- All data access via `utils/` modules — no direct file I/O in page files

### Security
- API keys via `python-dotenv`; never hardcode; `.env` is gitignored
- `CFBD_API_KEY`, `ODDS_API_KEY` in `.env`

### Error handling
- Return empty DataFrame on API failure, log the error
- Always guard `if df.empty` before rendering tables

---

## Export for Sports Picks Grid

Maintain `scripts/export_best_bets.py` to write `data_files/best_bets_today.json` with today's picks:
```json
{"meta": {"sport": "NCAAF", ...}, "bets": [...]}
```
Run: `python scripts/export_best_bets.py`
