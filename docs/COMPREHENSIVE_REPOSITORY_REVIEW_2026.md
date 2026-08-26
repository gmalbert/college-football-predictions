# Comprehensive repository review (August 2026)

> **Implementation review — 2026-08-24:** The remediation program described below is **substantially complete in code and artifacts**. A fresh strict audit passed all hard checks (9 pass, 0 fail), confirming grains, coverage, OOS artifact, manifests, and snapshot history. It also found active operational warnings: 87 legacy empty raw caches, stale best-bet/shadow exports (~319 hours), and a release decision of `hold`. The remaining work is operational maturity and prospective validation, not another generic model refactor.

## Executive conclusion

The repository has a useful end-to-end shape—CFBD ingestion, Parquet processing, feature engineering, four models, a Streamlit application, and scheduled automation—but the previously committed performance numbers were not decision-grade. The old feature artifact contained duplicate games, several season-final ratings were joined backward into earlier games, historical screens scored games with a model trained on those same games, and the spread sign convention was wrong in both edge calculation and ATS grading. Together these defects can make a model look far stronger than it is.

This review replaces those foundations with explicit grains, guarded joins, point-in-time transforms, chronological Elo, season walk-forward predictions, correct market math, release gates, model manifests, a data-quality page, and a unit-tested betting core. The old 64.9%/65.2% ATS claims must not be cited. Only metrics whose `evaluation_scope` is `walk_forward_season_oos` are valid.

Model v2.2 responds to that evidence instead of preserving a demonstrably inferior forecast. It recovers provider-level opening lines, constructs line consensus by median, and de-vigs moneylines within book before aggregation. On 2023–2025 OOS rows, overall win Brier is 0.1624 and the 2,492-game priced subset matches the 0.1736 market Brier. Margin RMSE is 17.57 overall and 15.198 on lined games, equal to the market. The market-state total correction improves lined RMSE from 15.830 to 15.793 and overall RMSE to 16.67.

More importantly, the new closing-time total-side classifier has OOS Brier 0.2492 versus 0.2500 and selects 185–138 (57.28%) at a 57.5% probability threshold. Side-specific validation retains only overs: 139–98 (58.65%) overall, with a 42–19 (68.85%) 2025 confirmation and 56.41% lower 95% Wilson bound. This model is deployed to a separate prospective shadow ledger; it is not an automated bet until 2026 timestamped prices demonstrate CLV.

The live feature artifact now contains 19,155 unique games: 17,517 completed 2021–2025 games and 1,638 scheduled 2026 games. The August 11 refresh also returned line payloads for 888 2026 games and wrote the first 440 normalized forward-only quote observations.

This is research software, not a guarantee of profit. A useful model must beat a strong market baseline after vig, latency, limits, line movement, and uncertainty—not merely predict winners.

## What was inspected

- Python source, Streamlit pages, configuration, scheduled workflow, caches, processed Parquets, feature matrix, serialized models, metrics, and export JSON.
- Row counts, primary-key uniqueness, join grains, label construction, missingness, date coverage, feature timing, model validation, settlement logic, and UI provenance.
- Comparable open-source data projects and primary literature listed in [PREDICTIVE_BETTING_RESEARCH.md](PREDICTIVE_BETTING_RESEARCH.md).

## Critical findings and disposition

| Severity | Finding | Consequence | Disposition |
|---|---|---|---|
| Critical | Feature matrix had 21,960 rows but only 17,517 distinct games | Some games were repeated many times and overweighted in training | Fixed with team-game deduplication, game-key rolling joins, guarded merges, and a hard feature contract |
| Critical | Season-final SP+, FPI/SRS, EPA/PPA/WEPA and related aggregates were available to earlier games | Temporal leakage | Excluded from active model feature lists unless a source supplies an auditable `available_at` snapshot |
| Critical | Model output is positive home margin, while a sportsbook home favorite is a negative spread; code subtracted them | Edge direction and size were wrong | Standardized `edge = model_home_margin + home_spread`; settlement and UI corrected |
| Critical | ATS record graded final-model predictions on training rows | Grossly optimistic performance | Replaced with expanding-season walk-forward predictions and a persisted OOS artifact |
| High | Rest was computed from a team's previous game only in the same home/away role | Incorrect fatigue feature | Rebuilt through a two-row-per-game team timeline |
| High | Empty two-byte JSON responses were treated as successful caches | Permanent cache poisoning | Empty caches are ignored and new empty responses are not written |
| High | Normal weekly runs reused current-season caches and skipped processed rebuilds | Live data could freeze | Current season now refreshes on every run; completed seasons remain cached; derived tables rebuild |
| High | Scheduled games were discarded because scores were null | No true future inference/export path | Games retain null labels until final; home/export paths select upcoming games only |
| High | Complete-case filtering reduced training to a small biased subset | Selection bias and poor early-season coverage | Native XGBoost missing handling or persisted median-plus-indicator imputation |
| High | Historical UI pages called the final fitted model | In-sample predictions appeared historical | `predict_for_display` uses OOS predictions for completed games and full-fit predictions only for future games |
| Medium | No market-price, push, void, devig, CLV, or quote-time abstraction | Unrealistic strategy evaluation | Added normalized market, settlement, EV, consensus, movement, dispersion and CLV utilities |
| Medium | No model/data lineage | Artifacts could not be reproduced or compared safely | Added schema/data fingerprints and per-model manifests |
| Medium | No automated tests or release audit | Regressions could ship silently | Added 25 core tests, strict audit CLI, CI test/audit gates, and Streamlit quality/model-evidence pages |
| High | Rebuilding from a partial current-season raw cache could replace processed history | A schedule refresh briefly exposed historical-partition loss risk | Processed builders now upsert refreshed partitions/keys and retain seasons absent from the current raw cache |

## Architecture assessment

### Ingestion and storage

Strengths are the clean raw/processed/features separation, local caching, Parquet outputs, and a single storage module. Weaknesses were unversioned mutable caches, swallowed upstream failures, no capture-time contract, no line history, and tracked empty responses. The new ingestion logic is substantially safer, but a production deployment should store immutable timestamped raw objects and explicit ingestion-run metadata in object storage.

### Feature engineering

The repository had ambitious feature breadth but confused “available somewhere in a season” with “known at prediction time.” The active models now use chronological Elo, preseason-known priors, rest, schedule context, strictly shifted game outcomes/team statistics, and a declared market snapshot. Optional contextual feature code covers travel, weather, quarterback status, roster continuity, pace, pressure, red-zone, special teams, rankings, and market quality, but these fields remain `NaN` until a timestamped provider adapter supplies them. This is deliberate: missing is safer than fabricated zero.

### Modeling and evaluation

XGBoost plus regularized linear baselines is reasonable for the dataset size. The challenger bake-off tested odds calibration, market-plus-structural logistic models, and Ridge/HistGradientBoosting/ExtraTrees residual corrections with chronological folds and nested shrinkage. None beat no-vig moneylines; spread corrections were economically indistinguishable from zero. Recovering opening/current market state materially improved the total target. Production therefore anchors win/spread to the market, uses structural fallbacks before prices exist, applies the total residual correction, and evaluates the specialized total-side classifier independently. See [TOTAL_MARKET_EDGE_SOLUTION.md](TOTAL_MARKET_EDGE_SOLUTION.md).

### Betting and risk

The original betting helpers conflated win probability with cover probability and did not model the price paid. The new core separates markets and sides, removes vig, evaluates EV, settles pushes, calculates CLV, and sizes a capped fractional-Kelly portfolio after shrinkage and an uncertainty haircut. No strategy should be promoted until prospective CLV is populated and passes its release gate.

### User interface

The app is approachable and the multipage organization is sound. The updated pages distinguish OOS historical predictions from future full-fit forecasts. The new Data Quality page exposes contract failures, missingness, stale exports, unsafe exploration columns, and artifact freshness. The feature dictionaries still include an extended candidate catalog, but now explicitly identify the active production inputs.

### Operations

The scheduled pipeline now runs tests, refreshes the live season, honors its `years` input, rebuilds all derived artifacts, trains, exports, runs the strict audit, and only then commits. Remaining production needs include atomic multi-artifact releases, retry/backoff with error provenance, immutable raw partitions, alerting, and a separate real-time odds job.

## Recommended delivery sequence

1. Run the implementation runbook and treat the first walk-forward metrics as the new baseline.
2. Collect timestamped multi-book odds and start measuring CLV prospectively.
3. Add injuries/QB status, weather forecasts, roster continuity, travel, and play/drive features only with `available_at` metadata.
4. Use the joint margin/total distribution to derive coherent win, cover, and total probabilities.
5. Run shadow predictions for a full season; promote only through release gates and stable CLV.

## Definition of done

A release is decision-grade only when all of the following are true:

- Game and team-game grains pass with zero duplicate keys.
- Every production feature is proven available no later than its prediction timestamp.
- All displayed historical predictions come from a fold in which that season was not trained.
- Metrics include market baselines, calibration, uncertainty, pushes, prices and sample sizes.
- The model has positive prospective CLV across a meaningful sample and acceptable drawdown.
- A model manifest links exact features, data fingerprint, training period, parameters, code SHA and prediction convention.
- Unit tests and `scripts/audit_pipeline.py --strict` pass.
