# Total-market edge solution (model v2.2)

> **Implementation review — 2026-08-24:** **Implemented in shadow mode.** The total-side model, OOS artifact, side gate, snapshot collector, export, workflow, and dashboard are present. It is **not complete as a deployable betting strategy**: current shadow output is stale and empty, no actual executable prices/CLV have been captured through a confirmation window, and promotion remains explicitly blocked.

## Outcome

The repository now has a specific, falsifiable path beyond market parity: predict the side of the college-football total at the final eligible pre-kickoff snapshot, using opening-to-current market behavior plus point-in-time team form. This is a separate model and lifecycle from win probability and spread forecasting.

The model remains in 2026 shadow deployment. It is not an automated betting claim. Retrospective accuracy is strong enough to justify prospective testing; timestamped executable prices and closing-line value are still required for promotion.

## What was wrong with the prior market layer

1. Raw 2021–2025 line cache files were empty even though CFBD could return historical provider records.
2. The processed layer averaged American odds, which is not valid probability aggregation.
3. Provider opening fields (`spreadOpen`, `overUnderOpen`) were discarded.
4. Cross-book disagreement and book count were discarded.
5. One global release decision hid the fact that totals contained signal while win/spread did not.
6. A point forecast was being converted to a side probability through a broad Normal approximation instead of training the side decision directly.

The new layer takes medians for line consensus, de-vigs each provider's two-sided moneyline before taking a probability consensus, and preserves current line, opening line, movement, dispersion, and book count.

## Recovered data

| Season | Games returned | Provider records |
|---:|---:|---:|
| 2021 | 887 | 3,541 |
| 2022 | 1,463 | 4,310 |
| 2023 | 1,416 | 3,072 |
| 2024 | 1,573 | 3,172 |
| 2025 | 1,597 | 3,345 |
| 2026 current pull | 888 payloads | 104 currently posted provider records |

The processed feature matrix has 4,435 games with opening spread movement and 4,462 with opening-to-current total movement.

## Model contract

- Target: `P(actual_total > current_market_total)`; pushes are excluded from binary training and grading.
- Model: median/indicator imputation, standardization, logistic regression with `C=0.005`.
- Features: current/open total, total movement, dispersion, book depth, current/open spread, spread movement, shifted scoring/yardage/turnover/third-down form, rest, and venue flag.
- Validation: expanding seasons—2021–2022 → 2023, through 2023 → 2024, through 2024 → 2025.
- Selection: chosen-side probability must be at least 57.5% (`abs(P(over)-0.5) >= 0.075`).
- Timing: only the final eligible snapshot within 12 hours of kickoff may emit a shadow signal.
- Price assumption in retrospective ROI: -110. Actual deployment must use captured executable prices.

## Walk-forward results

Probability Brier is 0.249154 versus 0.250000 for a 50/50 market-side baseline.

| Slice | Record | Win rate | Flat ROI at -110 | 95% Wilson interval |
|---|---:|---:|---:|---:|
| All selected sides | 185–138 | 57.28% | 9.34% | 51.83%–62.55% |
| Selected overs | 139–98 | 58.65% | 11.97% | 52.29%–64.73% |
| Selected unders | 46–40 | 53.49% | 2.11% | 43.02%–63.65% |
| 2025 selected overs | 42–19 | 68.85% | 31.45% | 56.41%–79.06% |

Over performance by held-out season was 65–56 (53.72%) in 2023, 32–23 (58.18%) in 2024, and 42–19 (68.85%) in 2025. Unders failed the side-stability gate because the 2023 result was 8–9 and its uncertainty remains wide. Consequently, only overs are eligible for the prospective shadow ledger.

The total point model also improved: like-for-like lined RMSE is 15.7932 versus 15.8300 for the market total. Spread corrections continue to shrink toward zero, so the system still abstains from spread recommendations.

## Why this is not yet a live betting system

The historical CFBD `spread`/`overUnder` values behave like final archived lines, but the archive does not provide every capture timestamp or prove that a quoted -110 price was executable when the signal was generated. Model and threshold development also inspected historical seasons; 2026 is the first genuinely prospective test.

Promotion therefore requires all of the following:

1. Append-only snapshots demonstrate the signal existed before kickoff.
2. At least 150 eligible prospective over signals, or a predeclared full-season stopping point.
3. Positive mean CLV and a positive-CLV rate whose uncertainty is reported.
4. Brier score remains below 0.25 and calibration does not materially drift.
5. Executable prices are captured; results are settled with pushes and actual odds.
6. Performance is not concentrated in one conference, week range, provider, or market-depth regime.
7. No threshold or feature changes are made during the confirmation window without resetting it.

## Operational implementation

- `utils/odds_ingestion.py`: provider normalization and robust current/open consensus.
- `utils/models.py`: fourth artifact, walk-forward total-side probabilities, side-specific gates, and manifests.
- `scripts/export_shadow_totals.py`: non-actionable closing-window signals.
- `scripts/snapshot_market.py --refresh-features`: one-call market refresh, feature rebuild, and shadow export.
- `.github/workflows/market_snapshots.yml`: additional game-day capture times.
- `pages/10_Total_Market_Signals.py`: evidence and live shadow state.

This is the solution to the model's earlier failure mode: stop asking generic public features to beat a closing market everywhere, preserve the information path from open to close, specialize where the data shows repeatable signal, and require the 2026 market to confirm it prospectively.
