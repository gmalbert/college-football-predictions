> **AI Onboarding Guide** — See also the project docs folder for detailed data and model documentation.

# College Football Predictions — Site Summary

## What This App Does

Full-stack Streamlit app for college football predictions using 5 years (2021–2025) of CFBD (College Football Data API) data. Predicts win probability, spread, and totals with XGBoost models; calculates value bets with Kelly-criterion bankroll sizing. In-sample ATS record: 64.9%.

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Run the app
streamlit run predictions.py
```

Data is refreshed weekly by GitHub Actions (`weekly_pipeline.yml`). For local data refresh: run the pipeline manually via `pages/6_Settings.py` controls or call `utils/cfbd_client.py` fetch functions directly.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit ≥1.55 (multi-page, 8 pages) |
| ML | XGBoost 2.0+ (win, spread, total) + ELO model + Ridge (totals backup) |
| Data source | CFBD v5 (College Football Data API) |
| Data storage | Parquet (primary) |
| Bankroll | Kelly Criterion (`utils/betting.py`) |
| Visualization | Plotly 5.18+ |

## Key Files

| File | Purpose |
|---|---|
| `predictions.py` | Home page — hero metrics, top picks, accuracy cards, dataset summary |
| `pages/1_Weekly_Predictions.py` | Spread/moneyline/total predictions by week with edge filtering |
| `pages/2_Value_Bets.py` | Value-bet table sorted by edge, bankroll simulator |
| `pages/3_Team_Explorer.py` | Team card, ELO history, radar chart, schedule |
| `pages/5_Model_Performance.py` | Brier score, calibration curve, feature importance |
| `pages/6_Settings.py` | API status, data refresh controls, model retraining triggers |
| `utils/cfbd_client.py` | CFBD v5 API client — fetches games, advanced stats, recruiting |
| `utils/feature_engine.py` | Feature matrix builder (86 features: EPA, turnovers, recruiting, form) |
| `utils/models.py` | XGBoost win/spread/total training, ELO model, batch predictions |
| `utils/betting.py` | Expected value, Kelly criterion, edge detection |

## Data Flow

1. **Fetch**: `utils/cfbd_client.py` → CFBD v5 (games, stats, team game stats, recruiting, ELO) → JSON cache
2. **Feature engineering**: `utils/feature_engine.py` → 86 features (rolling 5-game EPA, turnovers, explosiveness, havoc, recruiting rank diffs, SP+) → `feature_matrix.parquet` (21,932 rows × 86 cols)
3. **Training**: XGBoost win + spread + total models + ELO → `models/*.pkl`
4. **Predictions**: `predict_batch()` → `predictions_df` (pred_win_prob, pred_spread, pred_total)
5. **Upcoming fixtures**: ESPN API → merge with team stats → edge vs bookmaker odds
6. **UI**: Streamlit reads Parquet → 8-page dashboard

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `CFBD_API_KEY` | College Football Data API | Optional (free tier exists) |
| `ODDS_API_KEY` | The Odds API — historical odds for edge calibration | Optional |

## Critical Conventions

- Pipeline is **weekly** (not daily) — predictions may be up to 7 days stale
- All rolling features use 5-game windows with `shift(1)` to prevent leakage
- Feature set: 86 features including EPA/play, explosiveness, havoc rate, recruiting rank gaps, SP+ offense/defense, rest days
- `cfbfastr_integration_ideas.md` in docs describes play-by-play analytics upgrade (not yet implemented)
- Use `pathlib.Path` for all file paths

## Common Gotchas

- Moneyline model converts binary win probability to American odds — it is NOT a dedicated moneyline classifier
- Totals model uses Ridge regression (simpler than XGBoost) — upgrade documented but not yet implemented
- No injury/depth chart data: CFBD v5 does not expose this well
- Weekly pipeline means missed early-week line movements; consider daily ESPN fixture fetch as a supplement
