# Predictive betting research and open-source review

> **Implementation review — 2026-08-24:** The temporal validation, proper-score, calibration, market-baseline, uncertainty, and portfolio-control patterns described here are **implemented in code**. The operational research protocol is only **partially complete**: the repository has 440 forward snapshots, but does not yet have a meaningful prospective CLV sample, captured executable prices, or a locked full-season confirmation period. Treat all retrospective strategy results as research evidence, not a live-betting result.

## Open-source patterns worth adopting

The goal was not to copy model recipes; it was to identify durable data and operational patterns.

| Project | Useful pattern | Repository application |
|---|---|---|
| [cfbfastR](https://github.com/sportsdataverse/cfbfastR) and its [data repository](https://github.com/sportsdataverse/cfbfastR-cfb-data) | A broad, documented college-football data surface with game, play, drive, roster, ranking and betting datasets | Treat each source as a typed table with an explicit grain; maintain a data dictionary rather than a monolithic feature dump |
| [cfbfastR raw data](https://github.com/sportsdataverse/cfbfastR-cfb-raw) | Raw and enriched artifacts remain separate so processing can be rerun offline | Preserve immutable raw observations and rebuild processed/features from them |
| [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data) | Scheduled pipelines, release artifacts and update timestamps | Add freshness metadata, scheduled in-season refreshes and auditable release artifacts |
| [cfbd-python](https://github.com/CFBD/cfbd-python) | Generated typed client for CFBD endpoints | Keep the API wrapper thin and normalize SDK objects at the ingestion boundary |
| [nflverse play-by-play releases](https://github.com/nflverse/nflverse-pbp) | Versioned release assets decouple consumers from a live API | Publish immutable season partitions and a small current-season delta |
| [nfl_data_py](https://github.com/nflverse/nfl_data_py) | Local caching, selective columns, type/downcast utilities | Add column projection and typed loading when artifacts grow |
| [sportsdataverse-py](https://github.com/sportsdataverse/sportsdataverse-py) | Consistent Python loaders across sports datasets | Separate acquisition adapters from canonical schemas |
| [Feast](https://github.com/feast-dev/feast) | Point-in-time correct historical joins prevent future information from entering training rows | Every dynamic observation needs `available_at`; use backward as-of joins |
| [Pandera](https://github.com/unionai-oss/pandera) | DataFrame schema validation and statistical checks | The repository now has dependency-light contracts; Pandera remains a sensible scale-up path |
| [MLflow](https://github.com/mlflow/mlflow) | Tracking, evaluation, registry and champion/challenger lifecycle | Manifests, fingerprints, gates and promotion decisions mirror this lifecycle without a service dependency |
| [sdisorbo/cfb_spread_betting_model](https://github.com/sdisorbo/cfb_spread_betting_model) | XGBoost spread modeling, held-out-season evaluation, and side-specific performance reporting | Retain tree models as no-line fallbacks; report favorite/underdog and home/away slices rather than one headline ATS number |
| [Sports-Betting-Model](https://github.com/throwawayhub25/Sports-Betting-Model) | Reliability plots, comparison with sharp-book prices, and line-shopping workflow | Keep probability calibration, reference-book comparison, and executable price selection as distinct layers |

An instructive warning comes from a public [college-football model repository](https://github.com/blaizerlahman/CFB-Model) whose own README states that historical backtests used faulty data. The lesson is general: attractive results are not evidence until the underlying temporal and market contracts are independently checked.

## Research findings

### Optimize probability quality, not just classification accuracy

[Walsh and Joshi](https://arxiv.org/abs/2303.06021) show why calibration can matter more than raw accuracy for sports-betting decisions: stakes and expected value depend on the magnitude of probabilities, not only which side exceeds 50%. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) demonstrate that modern models can be miscalibrated and that post-hoc temperature scaling is a strong simple baseline. The implementation therefore records Brier score, log loss and expected calibration error, produces an OOS calibration artifact, and keeps calibration separable from discrimination.

### Use proper scoring rules and reliability diagnostics

The review by [Gneiting and Katzfuss](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-062713-085831) frames probabilistic forecasts around calibration and sharpness and discusses proper scoring rules. For this repository, Brier/log loss and interval coverage are primary forecast metrics; ATS hit rate is a downstream strategy metric that depends on selection and price.

### Respect temporal order

[Bergmeir, Hyndman and Koo](https://robjhyndman.com/publications/cv-time-series/) discuss cross-validation for time-series prediction. Sports seasons have rule, roster, coaching and market regime changes, so the implementation uses expanding season folds: train on earlier seasons and test on the next season. Rows are never randomly shuffled across the temporal boundary.

### Control backtest overfitting

[Bailey et al.](https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=2326253) formalize the probability of backtest overfitting when many strategies are tried and only the best is reported. Recommended controls are a logged experiment registry, locked thresholds, untouched seasons, explicit market baselines, bootstrap intervals, and prospective shadow deployment. Do not choose the final strategy by maximizing historical ROI over dozens of feature/edge/staking combinations.

### Quantify predictive uncertainty under dependence

[EnbPI](https://proceedings.mlr.press/v139/xu21h.html), [adaptive conformal inference](https://proceedings.mlr.press/v162/zaffran22a.html), and [sequential predictive conformal inference](https://proceedings.mlr.press/v202/xu23r.html) extend interval ideas to sequential or time-dependent settings. Recent work on [split conformal prediction under temporal dependence](https://proceedings.mlr.press/v313/barber26a.html) further emphasizes that exchangeability cannot be casually assumed. The repository includes a simple split-conformal primitive and interval scoring; production should calibrate it only on prior time blocks and monitor coverage by season/week.

### Treat the market as the benchmark

A recent [odds-only forecasting benchmark](https://arxiv.org/abs/2604.17194) reinforces a practical point: betting markets themselves are strong predictive features and baselines. Its favorite–longshot-bias adjustment motivates testing a simple logistic transform of no-vig odds before adding high-dimensional signals. Recent work on [market-calibrated sports forecasting](https://arxiv.org/abs/2605.16066) likewise treats calibration to the market as the dominant driver while retaining structural information only where it adds out-of-sample signal. All model metrics should show improvement relative to vig-removed moneyline probabilities, the implied home margin, and the market total. A model that fails to beat those baselines may still predict games reasonably but has not shown betting value.

The repository's first-stage bake-off supports that conclusion: odds-only recalibration and market-plus-structural win models failed to improve raw prices, and spread residual corrections shrank toward zero. After provider history was recovered, correct per-book de-vigging strengthened the priced win baseline to 0.1736 Brier. Win and spread therefore remain market anchors. Total-market state was different: adding opening/current movement, dispersion, and depth improved like-for-like total RMSE from 15.8300 to 15.7932 and enabled a separately calibrated side classifier. Full first-stage results are in [MODEL_CHALLENGER_BAKEOFF_2026.md](MODEL_CHALLENGER_BAKEOFF_2026.md); the implemented solution is in [TOTAL_MARKET_EDGE_SOLUTION.md](TOTAL_MARKET_EDGE_SOLUTION.md).

The next investigation recovered provider-level opening lines that had been left in empty caches. That changes the estimand: rather than pretending public team features will beat an archived closing line, the system can model information revealed between opening and the final pre-kickoff snapshot. [Baryla et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2149603) report early-season totals bias while finding no comparable sides bias, which is directionally consistent with this repository's result: spread residuals still collapse toward zero, while a regularized total-side classifier improves Brier score and produces a stable over-only retrospective slice. See [TOTAL_MARKET_EDGE_SOLUTION.md](TOTAL_MARKET_EDGE_SOLUTION.md).

### Separate forecasting from portfolio construction

Research combining [sports forecasting with portfolio optimization](https://arxiv.org/abs/2307.13807) highlights that bet selection and allocation are separate problems. Kelly's original growth-optimal formulation is described in [Kelly (1956)](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1956.tb03809.x). Because estimated edges are noisy and bets are correlated, this repository implements probability shrinkage, uncertainty haircuts, fractional Kelly, per-bet/per-game/per-team/per-slate caps, and correlation penalties.

## Practical research protocol

1. Freeze a prediction timestamp and join only information with `available_at <= prediction_time`.
2. Train on earlier seasons; predict the next season once; persist those predictions.
3. Compare probability, margin and total forecasts with market baselines.
4. Calibrate only from earlier OOS folds, never from the season being reported.
5. Create ledgers using the actual line and price available at prediction time.
6. Report all eligible bets, pushes, voids, stake, profit, ROI, CLV and drawdown.
7. Bootstrap time clusters such as season-week, not individual bets as if independent.
8. Lock the model and thresholds before prospective shadow testing.
9. Promote on probability quality and CLV; treat short-horizon ROI as noisy.
