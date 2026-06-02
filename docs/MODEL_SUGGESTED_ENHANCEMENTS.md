# Tailgate Edge — Model Suggested Enhancements

## Priority 1: Elo Model Improvements

### Margin of Victory Weighting
- Current Elo update uses only W/L binary outcome. Incorporate **margin of victory** scaling: a 35-point win should update Elo more than a 3-point overtime win.
- Use `delta = min(abs(home_score - away_score) / 14, 2.5)` as a multiplier.

### Conference Strength Adjustment
- Power 4 teams defeating Group of 5 teams inflate Elo of the Power 4 incorrectly.
- Apply a **strength-of-schedule multiplier** (SOS) from CFBD API to scale K when teams are from different conferences.

### Preseason Reset Strategy
- Currently regressing 1/3 toward 1500. Consider **recruiting rank-weighted regression**: teams with top-10 recruiting classes should regress less aggressively.

## Priority 2: Feature Engineering

### Scoring Efficiency
- Add `points_per_drive` and `plays_per_drive` (from CFBD drive data) as features. More informative than raw points.

### Turnover Margin
- Turnovers are high-variance but predictive in college football. Add rolling 4-game `turnover_margin`.

### Home Crowd Effect
- College home crowds have a measurable effect (larger at larger stadiums). Encode `home_stadium_capacity` as a numeric feature.

### Third Down Conversion Rate
- Available from CFBD stats. Add rolling 3-game home/away third-down conversion differential.

## Priority 3: Model Expansion

### XGBoost Upgrade
- Current logistic regression baseline. Train an XGBoost model on the engineered feature set.
- Use `GroupKFold` by season to prevent data leakage.

### Spread Coverage Model
- Separate model targeting `covered_spread` (binary). This has direct betting application.

### Total Points Model
- Poisson regression for expected total points; calibrate against over/under market.

## Priority 4: Calibration

- Add calibration curve visualisation on Model Performance page.
- Apply isotonic regression post-calibration.
