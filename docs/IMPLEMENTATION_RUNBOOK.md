# Implementation and operations runbook

> **Implementation review — 2026-08-24:** This runbook's pipeline, audit, manifests, and snapshot commands are **implemented**. A strict audit completed with no failures during this review; it reported warning-only remediation work (legacy empty caches, stale exports, and a `hold` release). The local unit-test suite could not be executed in this desktop runtime because Windows application control blocked SciPy's compiled DLL—not because of a test assertion failure. CI remains the authoritative test environment.

## Local validation

From the repository root on Windows:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
venv\Scripts\python.exe scripts\audit_pipeline.py
```

The first audit may report the old duplicate feature/model artifacts. That is expected until the rebuild below completes.

## Safe rebuild from existing caches

```powershell
venv\Scripts\python.exe -m utils.fetch_historical
venv\Scripts\python.exe -m utils.feature_engine --force
venv\Scripts\python.exe -m utils.models --force
venv\Scripts\python.exe scripts\export_best_bets.py
venv\Scripts\python.exe scripts\export_shadow_totals.py
venv\Scripts\python.exe scripts\audit_pipeline.py --strict --output data_files\audit_report.json
```

Normal ingestion reuses non-empty caches for completed seasons, refreshes the current season, and rebuilds processed tables. Use `--force` only when source history or parsing changed and the API budget permits a full repull. Use `--years 2022,2023,2024,2025,2026` to select an explicit range.

## Artifact checks

- `processed/games.parquet`: one row per game; scheduled games retained with null labels.
- `processed/team_game_stats.parquet`: one row per game/team.
- `features/feature_matrix.parquet`: one row per game.
- `features/model_backtest.parquet`: OOS columns populated only for held-out seasons.
- `models/model_metrics.json`: `evaluation_scope` must be `walk_forward_season_oos`.
- `models/*_manifest.json`: feature list, hashes, training period and prediction convention.
- `best_bets_today.json`: only upcoming games, UTC generation timestamp and model version.
- `shadow_total_signals.json`: final-window research signals; never interpret these as promoted bets.

## Reading the metrics

Do not compare the new numbers with the old 64.9%/65.2% ATS claim; that claim combined leakage, duplicate weighting, sign errors and in-sample grading. Read model metrics against their baselines:

- Win: lower Brier/log loss and positive Brier skill versus vig-removed moneylines.
- Spread: RMSE/MAE versus the implied market margin (`-home_spread`).
- Total: RMSE/MAE versus the market total.
- Calibration: low ECE and a reliability curve near the diagonal.
- Strategy: price-aware ROI with pushes, uncertainty intervals, drawdown and prospective CLV.

ATS hit rate over every lined game is a diagnostic, not a deployable strategy. Thresholded strategy results belong in a bet ledger.

## Release process

1. Tests pass.
2. Ingestion succeeds; empty responses do not replace valid caches.
3. Processed and feature contracts pass.
4. Walk-forward training completes and persists OOS predictions/manifests.
5. Release gates are evaluated. Missing prospective CLV should keep the artifact a challenger.
6. Export is generated only from upcoming games.
7. Strict audit passes before committing generated artifacts.
8. Run a prospective shadow period before risking money.

## Adding a new feature source

1. Define source grain, units, event time, `available_at`, timezone and revision behavior.
2. Save raw observations immutably; distinguish empty success from an error.
3. Normalize to a canonical processed table and validate its unique key.
4. Join backward as of the prediction timestamp with `utils.temporal.point_in_time_join`.
5. Add a coverage indicator and leave missing values null.
6. Add tests proving a later observation cannot enter an earlier game.
7. Register the feature, bump feature/model version and regenerate manifests.
8. Compare on fixed walk-forward folds against the current champion and market baseline.

## Timestamped odds integration

The repository contains quote selection, consensus, movement, dispersion, devig, EV, settlement and CLV code. A real feed should emit the `line_snapshots` schema from [DATA_MODEL_V2.md](DATA_MODEL_V2.md). Never call a mutable “current line” the closing line. Closing is the final eligible pre-kickoff snapshot, and every backtest selection must use the price actually available at its own prediction timestamp.

Start or test the forward-only CFBD collector with:

```powershell
venv\Scripts\python.exe scripts\snapshot_market.py --season 2026
```

For a game-day refresh that also rebuilds the current consensus/features and emits the closing-time shadow ledger:

```powershell
venv\Scripts\python.exe scripts\snapshot_market.py --season 2026 --refresh-features
```

The scheduled daily workflow appends idempotently to `processed/line_snapshots.parquet`. It cannot backfill historical observations that were never captured.

## Incident handling

- Duplicate-key failure: stop; inspect upstream grain and merge cardinality. Do not `drop_duplicates` blindly in the feature matrix.
- Empty current-season API response: retain the previous valid cache, log the source error, and do not publish a new release.
- Missing OOS artifact: historical model charts must remain unavailable; never substitute final-model predictions.
- Drift alert: hold promotion, compare source coverage/units and rerun shadow evaluation.
- Stale export: show no bets rather than yesterday's recommendations.
