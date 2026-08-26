# Current-model backtest audit (2026)

## Verdict

The saved artifact reports win-model Brier 0.1594 and log loss 0.4839 on 5,100 samples; spread RMSE 13.536 on 5,100 and total RMSE 15.255 on 4,461. It also claims 3,325 ATS wins and 1,775 losses (65.2%). The artifact contains no seasons, train/test boundaries, market lines/odds, pushes, model version, or baseline. A 65.2% ATS rate over 5,100 games would be extraordinary and is inconsistent with efficient markets; without a temporal contract it should be treated as leakage/target-definition evidence, not a betting edge.

## Changes justified by the result

1. Audit whether spread labels or final-score-derived features enter predictors, and whether lines are joined to the correct team/sign.
2. Use leave-one-season-out/rolling-season validation, with week-level cutoffs and no end-of-season aggregates.
3. Compare win probabilities, margins, and totals to de-vigged closing markets and simple Elo/SP+ style baselines.
4. Model conference/team partial pooling, returning production, portal/coaching change, quarterback status, garbage time, weather, travel, and FCS games explicitly.

## Betting strategy decision

- **Moneyline:** probability metrics are promising but unauditable; paper-only.
- **Spread:** invalidate the 65.2% claim until leakage/sign/settlement tests pass.
- **Totals/team totals:** 15.3-point RMSE needs line-specific calibration.
- **Props/live:** insufficient timestamped player/state data.
- **Futures/parlays:** require season simulation and correlated outcomes.
- **Staking:** no Kelly; flat paper stakes.

## Release gate

Untouched recent season, reproducible feature-as-of audit, actual selection/closing odds, 500+ settled bets, and ATS/ROI confidence intervals with all threshold and model-search choices disclosed.
