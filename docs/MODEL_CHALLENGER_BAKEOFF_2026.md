# Model challenger bake-off (August 2026)

> **Implementation review — 2026-08-24:** **Completed as a reproducible research artifact.** The candidate implementations, benchmark script, and v2.1/v2.2 model path exist. The decision remains **hold/shadow**, not promoted: the current manifest shows zero win-market skill, market-parity spread performance, and no prospective CLV evidence. Numbers in this document are historical v2.1 evidence; use `data_files/models/model_metrics.json` for the current v2.2 artifact.

> This document records the v2.1 first-stage bake-off. Provider-level opening-line recovery subsequently produced v2.2 and the specialized total-side solution documented in [TOTAL_MARKET_EDGE_SOLUTION.md](TOTAL_MARKET_EDGE_SOLUTION.md). The tables below remain useful as evidence that generic win/spread complexity did not beat the market.

## Decision

Model v2.1 promotes a market-anchored architecture, not the most complex candidate. Win and margin use the market consensus when present and preserve structural models as pre-market fallbacks. Totals use a small Ridge correction to the market because that was the only candidate family with a meaningful like-for-like OOS improvement. The release remains `hold` until win/spread add value beyond parity and prospective CLV is positive.

## Evaluation contract

- Outer folds: train 2021–2022/test 2023; train through 2023/test 2024; train through 2024/test 2025.
- Comparisons use exactly the same priced/lined rows for model and market.
- Residual-correction shrinkage is learned from an inner expanding-season walk-forward loop inside each outer training fold.
- No current test-season outcomes choose a fold's model, correction size, or calibration.
- Candidate breadth is disclosed to avoid presenting the winner as if it were the only model tried.

## Results

### Home-win probability (2,487 priced OOS games)

| Candidate | Brier | Difference vs raw no-vig market |
|---|---:|---:|
| Raw no-vig market | 0.176040 | reference |
| Odds-only logistic recalibration | 0.176198 | +0.000158 worse |
| Market + Elo, best tested regularization | 0.176364 | +0.000323 worse |
| Market + core structural features, best tested | 0.177804 | +0.001764 worse |
| Market + all production structural features, best tested | 0.178519 | +0.002479 worse |
| Previous v2.0 structural model | 0.189378 | +0.013338 worse |

Production uses raw no-vig probability on priced games and the chronological structural classifier where moneylines are missing. This is an improvement over v2.0, but it is market parity—not evidence of a bettable win-probability edge.

### Home margin (4,567 lined OOS games)

| Candidate | RMSE | Difference vs market |
|---|---:|---:|
| Implied market margin | 15.199748 | reference |
| Nested-shrunk Ridge residual | 15.199706 | -0.000042 |
| Nested-shrunk HistGradientBoosting residual | 15.199748 | 0.000000 |
| Nested-shrunk ExtraTrees residual | 15.199138 | -0.000611 |
| Previous v2.0 structural model | 15.4948 | about +0.295 worse |

The apparent residual improvements are too small and concentrated in later folds to justify added complexity. Production uses the implied market margin when lined. With a one-point minimum edge, all market-parity rows are abstentions.

### Game total (4,558 lined OOS games)

| Candidate | RMSE | Difference vs market |
|---|---:|---:|
| Market total | 15.829390 | reference |
| Nested-shrunk Ridge residual | 15.807419 | -0.021971 |
| Nested-shrunk HistGradientBoosting residual | 15.815893 | -0.013497 |
| Nested-shrunk ExtraTrees residual | 15.806169 | -0.023221 |
| Previous v2.0 structural model | 16.059 | about +0.230 worse |

Ridge was promoted instead of ExtraTrees: their scores are nearly identical, Ridge is lower variance and easier to audit, and selecting the nominal winner after trying several models would overstate the evidence. Fold shrinkage rose from 0.014 (2023) to 0.217 (2024) and 0.332 (2025); the final full-history estimate is 0.369.

## Implementation

- `utils/challenger_models.py`: candidate estimators, inner/outer temporal shrinkage, and market/fallback composite artifacts.
- `scripts/benchmark_challengers.py`: reproducible candidate table.
- `utils/models.py`: v2.1 training, persisted OOS predictions, abstention-aware ATS diagnostic, and release gates.
- `utils/feature_engine.py`: canonical `market_home_prob` feature.

## What would change the decision

Collect a full prospective season of timestamped multi-book prices, lock prediction times, and measure CLV. Promote a structural win/spread correction only if it improves proper scores on identical rows, remains stable by season and market regime, and produces positive prospective CLV after price/limit constraints. Until then, market parity plus abstention is the honest champion.
