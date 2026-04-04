# college-football-predictions

Data-driven college football predictions and value-betting analytics — a full-stack Streamlit app backed by 5 years of CFBD historical data, XGBoost models, and Kelly-criterion bankroll sizing.

## Overview

- **Win probability** — XGBoost binary classifier (Brier score: 0.160)
- **Spread projections** — XGBoost regressor (RMSE: 13.80 pts)
- **Totals projections** — Ridge regressor (RMSE: 15.14 pts)
- **ATS record** (in-sample, FBS vs FBS): **64.9%** (3,297 W / 1,783 L)
- **Value bets** — spread, total, and moneyline edge alerts with Kelly sizing
- **5 seasons of data** — 2021–2025, 17,517 games, ~46 API calls total
- **Feature matrix** — 21,932 rows × 86 features including rest days, rolling team-game stats, and recruiting rank diffs

## Recent updates

- Implemented feature and model roadmap from `docs/features.md`, `docs/data_pipeline.md`, and `docs/models.md`.
- Added weekly CFBD `team_game_stats` ingestion and `team_game_stats.parquet` to support rolling 5-game stats, turnovers, rushing/passing EPA, penalties, and special-teams metrics.
- Expanded the feature engine with new diffs for SP+ offense/defense, rushing/passing EPA, explosiveness, havoc, rest advantage, and recruiting rank.
- Added a standalone `EloModel` and improved XGBoost training with log-loss tracking, higher max depth, and stronger early stopping.
- Added O/U predictions and moneyline recommendations to the prediction pages.
- Added `.gitignore` exclusions for generated data directories and common Python/Streamlit artifacts.
- Settings page is hidden automatically on Streamlit Cloud deployments.

## Pages

| Page | Description |
|---|---|
| 🏠 Home (`predictions.py`) | Summary cards — top picks, model accuracy, dataset stats |
| 📊 Weekly Predictions | Game cards for any week/season with spread, win prob, and edge |
| 💰 Value Bets | Value-bet table sorted by edge, bankroll simulator |
| 🏟 Team Explorer | Team card, Elo history chart, radar chart, schedule |
| 📈 Historical Analysis | Season trends, head-to-head lookup, conference power rankings |
| 🎯 Model Performance | Brier score, calibration curve, feature importance, ATS by week |
| ⚙ Settings | API status, data refresh controls, model retraining |

## Data pipeline

Data is fetched from the [College Football Data API](https://api.collegefootballdata.com/) (CFBD v5) and cached as JSON so repeat runs never hit the network. Processed Parquet tables are built from the cache.

```
data_files/
  raw/              JSON cache (one file per endpoint per year)
  processed/        Parquet tables — games, lines, ratings, advanced_stats,
                    recruiting, elo_ratings, team_game_stats
  features/         feature_matrix.parquet  (21,932 rows × 86 columns)
  models/           win_prob_model.joblib, spread_model.joblib,
                    total_model.joblib, model_metrics.json
```

API call budget: **~46 calls** for a full 5-year pull (one call per endpoint per season). All data is re-used from disk on subsequent runs.

> Use `python -m utils.fetch_historical --force` to rebuild all raw data and `team_game_stats.parquet` when the pipeline changes.

## Feature engineering

| Feature | Source |
|---|---|
| `elo_diff` | CFBD end-of-season Elo differential |
| `sp_plus_diff` | SP+ rating differential (FBS only) |
| `off_epa_diff` | Offensive EPA/play differential |
| `def_epa_diff` | Defensive EPA/play differential (inverted) |
| `recruiting_diff` | 3-year rolling recruiting points differential |
| `talent_diff` | Composite talent score differential |
| `home_flag` | 1.0 = home game, 0.5 = neutral site |
| `conference_game` | Boolean — same-conference matchup |
| `market_spread` | Closing spread (spread/total models only) |

## Getting started

1. **Install dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```

2. **Set your CFBD API key** in `.streamlit/secrets.toml`:
   ```toml
   [cfbd]
   api_key = "YOUR_CFBD_API_KEY"
   ```
   Or set the environment variable `CFBD_API_KEY` (also accepts `CBBD_API_KEY`).
   Get a free key at [collegefootballdata.com](https://collegefootballdata.com).

3. **Pull historical data** (one-time, ~46 API calls):
   ```bash
   python -m utils.fetch_historical
   ```

4. **Build feature matrix**:
   ```bash
   python -c "from utils.feature_engine import build_feature_matrix; build_feature_matrix(force=True)"
   ```

5. **Train models**:
   ```bash
   python -c "from utils.models import train_all; train_all(force=True)"
   ```

6. **Run the app**:
   ```bash
   streamlit run predictions.py --server.port 8502
   ```
   Or use the ⚙ Settings page to trigger data refresh and model retraining from the UI.

## Repository structure

```
predictions.py              Home page
pages/
  1_Weekly_Predictions.py
  2_Value_Bets.py
  3_Team_Explorer.py
  4_Historical_Analysis.py
  5_Model_Performance.py
  6_Settings.py
utils/
  cfbd_client.py            CFBD API v5 wrapper (access_token auth)
  fetch_historical.py       5-year data pull script
  feature_engine.py         Feature matrix builder
  models.py                 XGBoost/Ridge train, persist, inference
  betting.py                Edge detection, Kelly criterion
  elo.py                    Custom Elo model
  espn_client.py            ESPN public API wrapper
  config.py                 Secrets / env-var loading
  storage.py                Parquet read/write helpers
  logger.py                 Structured logging
  ui_components.py          Shared Streamlit sidebar + cards
data_files/                 Raw JSON cache + processed Parquets + models
docs/                       Planning docs (layout, features, models, etc.)
.streamlit/
  config.toml               Dark theme, server settings
  secrets.toml              API key (not committed)
requirements.txt
```

## Streamlit notes

- `use_container_width=True` is deprecated — use `width="stretch"` for responsive full-width components and `width="content"` for fixed-width content.
- Use `st.columns` and explicit layout components for responsive design.

## Model notes

- Models are trained on FBS-vs-FBS games with complete SP+, EPA, and Elo data (~3,933 games).
- Elo ratings are CFBD end-of-season values used as a season-level strength signal.
- The spread model requires a closing market line as an input feature (`market_spread`).
- All models are saved with `joblib` and versioned by a `model_metrics.json` manifest.

