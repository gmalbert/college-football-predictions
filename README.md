# College football predictions

A Streamlit research application for college-football forecasting, market comparison, and leakage-safe betting analysis. It ingests CFBD data, builds point-in-time features, trains win/margin/total models, persists season walk-forward predictions, and exposes model and data-quality dashboards.

## Current validated baseline

The August 2026 rebuild contains 19,155 unique games: 17,517 completed games across 2021–2025 plus 1,638 scheduled 2026 games. Metrics below are expanding-season walk-forward results for 2023–2025 (11,358 held-out games), not training-set scores.

| Target | Model | OOS result | Market baseline | Conclusion |
|---|---|---:|---:|---|
| Home win probability | Per-book no-vig consensus + XGBoost fallback | Overall Brier 0.1624; priced-subset Brier 0.1736 | Priced-subset Brier 0.1736 | Market parity, not an edge |
| Home margin | Median market anchor + XGBoost fallback | Overall RMSE 17.57; lined-subset RMSE 15.198 | Lined-subset RMSE 15.198 | Market parity, not an edge |
| Game total | Market-state residual Ridge + fallback | Overall RMSE 16.67; lined-subset RMSE 15.793 | Lined-subset RMSE 15.830 | Positive OOS improvement |
| Closing-time over classifier | Regularized logistic model | Brier 0.2492; high-confidence overs 139–98 (58.65%) | Coin-flip Brier 0.2500; -110 break-even 52.38% | Retrospective edge; 2026 shadow deployment |
| Actionable spread decisions | Minimum one-point edge | 0 bets; 4,567 abstentions | Market anchor has zero point edge | Correct behavior is to abstain |

The global model release remains **hold**, but v2.2 has a concrete target-specific solution in shadow deployment. The closing-time total classifier produced 185–138 (57.28%) across all high-confidence sides and 139–98 (58.65%) for the only side that passed stability gates: overs. Its 2025 confirmation was 42–19 (68.85%) with a 95% Wilson lower bound of 56.41%. These are retrospective results assuming -110; the over signal remains shadow-only until timestamped 2026 prices demonstrate prospective CLV. The former 64.9%/65.2% ATS figures were invalid and must not be cited.

## What changed

- One row per game and one row per team/game are enforced by executable data contracts.
- Team rolling features are shifted before aggregation and joined by game/team keys.
- Pregame Elo walks forward chronologically; season-final ratings are excluded from active model inputs.
- Rest follows a team's previous game across both home and away roles.
- Home spread uses sportsbook notation (`-7` means home favored by seven); model margin is `home - away`; edge is their sum.
- Models use expanding-season validation and persist `model_backtest.parquet` for honest historical pages.
- Model v2.1 uses no-vig/margin market anchors when a current price exists, structural fallbacks before lines appear, and nested OOS shrinkage for total-market residuals.
- Model v2.2 recovers provider-level opening history, uses median line consensus, de-vigs within each book, and models opening-to-current movement, dispersion, and book depth.
- A fourth artifact estimates `P(over current total)` under a final-pre-kickoff prediction contract; only ≥57.5% over signals enter the shadow ledger.
- Spread evaluation abstains when the forecast does not differ from the market by at least one point; market parity is never mislabeled as a bet.
- Market utilities cover odds conversion, vig removal, EV, quote consensus/as-of selection, movement, CLV, pushes and voids.
- A forward-only CFBD adapter normalizes provider quotes and appends a daily immutable market-snapshot table for future movement/CLV analysis.
- The 2026 pull currently includes 1,638 scheduled games, 888 CFBD game-line payloads, and an initial normalized snapshot of 440 quotes.
- Risk utilities add shrinkage, uncertainty haircuts, fractional Kelly, concentration caps and correlation penalties.
- Model manifests record exact features, data/schema fingerprints, training period, metrics and prediction contracts.
- CI runs predictive-core tests and a strict data/model audit before publishing artifacts.
- The app includes a Data Quality page and distinguishes historical OOS predictions from future full-fit forecasts.
- A Total Market Signals page exposes the separate OOS record, side-specific gates, and prospective shadow file.

## Pages

| Page | Purpose |
|---|---|
| Home | Upcoming model edges, validated OOS metrics and dataset summary |
| Weekly Predictions | Selected slate; historical games use saved OOS predictions |
| Value Bets | Price/line disagreement and strategy visualization |
| Team Explorer | Team history and matchup detail |
| Historical Analysis | Results, trends and OOS ATS diagnostics |
| Model Performance | OOS calibration, baselines, feature importance and ATS by week |
| Data Quality | Contracts, duplication, freshness, missingness and leakage warnings |
| Total Market Signals | Closing-time total probability, OOS selections and 2026 shadow signals |
| Settings | Data refresh, feature build and model training |

## Data flow

```text
CFBD/API caches -> canonical processed Parquets -> point-in-time feature matrix
               -> expanding-season folds -> OOS predictions + final future model
               -> market/settlement/risk layer -> Streamlit and JSON export
```

Generated layout:

```text
data_files/
  raw/          source caches; empty responses are never accepted as valid
  processed/    games, lines, team-game stats, ratings and optional sources
  features/     feature_matrix.parquet and model_backtest.parquet
  models/       joblib models, model_metrics.json and per-model manifests
  audit_report.json
  best_bets_today.json
```

## Getting started

1. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Set `CFBD_API_KEY` or add it under `[cfbd]` in `.streamlit/secrets.toml`.

3. Run the pipeline:

   ```powershell
   venv\Scripts\python.exe -m utils.fetch_historical
   venv\Scripts\python.exe -m utils.feature_engine --force
   venv\Scripts\python.exe -m utils.models --force
   venv\Scripts\python.exe scripts\export_best_bets.py
   venv\Scripts\python.exe scripts\audit_pipeline.py --strict
   ```

4. Start the app:

   ```powershell
   streamlit run predictions.py --server.port 8502
   ```

5. Run tests:

   ```powershell
   venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

Normal ingestion reuses completed-season caches, refreshes the current season, retains scheduled games with null labels, and rebuilds derived tables. `--force` repulls all selected seasons; `--years 2022,2023,2024,2025,2026` selects an explicit range.

## Documentation

- [Comprehensive repository review](docs/COMPREHENSIVE_REPOSITORY_REVIEW_2026.md)
- [61-feature/change catalog](docs/FEATURE_CHANGE_CATALOG.md)
- [Data model v2](docs/DATA_MODEL_V2.md)
- [Predictive-betting research](docs/PREDICTIVE_BETTING_RESEARCH.md)
- [Implementation runbook](docs/IMPLEMENTATION_RUNBOOK.md)

## Important limitations

- The committed raw cache still contains 92 legacy empty JSON files. New ingestion ignores them, but a successful authenticated repull is needed to populate those optional sources.
- Season-level exploration columns remain in the feature artifact for UI/research compatibility but are excluded from active model feature lists.
- Genuine line movement and CLV require immutable, timestamped multi-book odds snapshots; the code/schema exists, but the current historical cache cannot recreate observations never collected.
- `scripts/snapshot_market.py` starts that collection prospectively; CFBD may omit spread/total prices, so a richer paid odds feed is still desirable.
- Weather, injuries, quarterback status, travel and roster context must be timestamped as they were known before the prediction. Observed postgame values are not valid substitutes.
- A good forecast is not automatically a profitable bet. Prices, limits, latency, correlation, uncertainty and responsible bankroll constraints matter.
