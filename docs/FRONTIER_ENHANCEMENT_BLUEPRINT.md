# Frontier Enhancement Blueprint

Current docs cover broad data sources, cfbfastR ideas, features, models (including graph concepts), calibration, betting strategy, deployment, and UI. These additions address modern roster churn, possession quality, and cross-level transfer.

## Roster and scheme continuity

Build time-bounded player/coordinator/team graphs from recruiting, transfer portal, returning production, snap counts, and staff changes. Translate player priors by source conference and role; shrink aggressively for FCS transfers and freshmen.

```python
def continuity(snaps_returning, coordinator_same, qb_same, ol_starts_returning):
    return (0.35 * snaps_returning + 0.20 * coordinator_same
            + 0.25 * qb_same + 0.20 * np.tanh(ol_starts_returning / 60))
```

## Possession-quality model

Use drive/series state, success rate, explosiveness, field position, pace, garbage-time probability, and special teams. Generate margin/total jointly so moneyline, spread, and total outputs remain coherent. Learn team strength hierarchically across conference and season.

## Data additions

- Portal/recruiting/coaching changes with effective dates.
- Participation and depth-chart observations with timestamps.
- Drive/play data, officiating crew, weather, altitude, travel, and academic-calendar breaks.
- Market prices from open through kickoff.
- Explicit lower-division and bowl opt-out coverage.

## Product additions

- Roster continuity and coordinator-change cards.
- “Garbage time on/off” metric explorer.
- Depth-chart scenario comparison.
- Rivalry/bowl/playoff context presented as features only when validated.
- Source confidence badge for small-program coverage.

## Gates

Validate forward by week/season and hold out promoted/FBS-transition programs. Report log loss, calibration, CRPS/MAE, CLV, and performance by conference, roster churn, quarterback uncertainty, weather, and favorite size. Compare to SP+/Elo-style and market baselines. Enforce `available_at <= prediction_time` for every roster/news field.
