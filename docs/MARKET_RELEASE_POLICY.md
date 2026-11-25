# NCAAF market release policy

This policy is locked for the current prospective validation window. A market
stays research-only until its evidence gates pass. Pick counts, point-edge
cutoffs, and retrospective records are not promotion criteria.

## Price contract

- Every executable candidate must use a pregame quote captured within 24 hours
  of export, at the exact selected line, with a paired opposing quote from the
  same sportsbook and provider captured within 30 minutes of it.
- Report the model probability, the raw implied probability of the displayed
  sportsbook price, expected value at that price, and the same-book de-vigged
  consensus separately. Never combine a consensus probability with a synthetic
  price and describe them as one quote.
- Store sportsbook, line, odds, and capture time with each candidate. Missing,
  stale, unmatched, or live quotes mean abstain.
- The export's `edge` is model probability minus the raw implied probability of
  the displayed price. `market_edge` is model probability minus de-vigged
  consensus. Neither is a validated edge unless the release gates pass.
- `market_quote_count` counts sportsbooks with a complete pair. The feed
  includes both prices and capture times for the selected sportsbook.
- Moneyline scoring uses the same fresh paired consensus used to measure the
  market edge. A stale feature snapshot cannot become a pricing signal.
- CLV compares the entry quote to a strictly later pre-kickoff, non-live quote
  from the same sportsbook and provider. Without a later observation the row
  has no CLV evidence. Total CLV is side-aware: line movement toward the chosen
  over or under is bettor-positive.
- The v2.3 validation window freezes the first qualifying priced signal for
  each strategy, game, and market. Later forecasts cannot replace the entry or
  select its other side. Prior strategy versions remain in the research ledger
  outside the v2.3 release sample.

## Market promotion

- **Moneyline:** the market probability correction uses only earlier seasons to
  fit the calibration and its shrinkage. Walk-forward Brier must beat the
  de-vigged market baseline on the same games. Calibration and the prospective
  CLV rule must pass.
- **Spread:** the market-adjusted forecast used by the exporter is the forecast
  graded on walk-forward margin RMSE. It must beat the market on the same games
  and the prospective CLV rule must pass.
- **Totals:** the total-market baseline and the prospective CLV rule must pass.
  The closing-time over-only candidate can record research signals when its
  side-level selection criteria and exact-line price rule pass. Promotion also
  requires the totals probability model to beat its retrospective baseline.
- **Prospective CLV:** require at least 150 settled prospective signals across
  at least eight distinct game weeks for the market/eligible side, with the
  week-clustered 95% confidence interval for mean
  bettor-positive CLV entirely above zero and the interval for the positive-CLV
  rate entirely above 50%.
- Do not tune the model, threshold, side eligibility, price rule, or selection
  contract during this window. Any such change starts a new validation window.

The aggregate model release remains on hold if any required gate fails. A
market may be described as shadow only when its own retrospective gates pass;
shadow is not validated or actionable.

## Display policy

- A held release is visibly marked `RESEARCH ONLY` on every prediction surface.
- The public best-bets feed emits an empty list during a release hold.
  Totals stay in the separate research feed until an export-time totals
  probability strategy passes validation.
- Point differences may be shown with their units, but point-based confidence
  tiers and wager language are suppressed.
- The point-only Value Bets view does not display records as betting evidence
  and does not offer bankroll simulation.
- Refresh generated feeds and static snapshots after changing this policy or
  release metadata; stale artifacts are not current publication state.
