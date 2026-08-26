# Data model v2: grains, time and market contracts

> **Implementation review — 2026-08-24:** Core contracts are **implemented and artifact-verified**: `games`, `team_game_stats`, `feature_matrix`, `model_backtest`, model manifests, and forward-only `line_snapshots` all exist and pass the strict audit with no failures. `feature_observations` and `bet_ledger` remain **target schemas**: the temporal join and betting/settlement primitives exist, but no general timestamped contextual-source store or production placed-bet ledger has been populated. Historical intermediate quote history and price-complete CLV are also still unavailable.

## Design rules

Every table has one declared grain. Every mutable fact has both an event time and, when relevant, an `available_at` or `captured_at` time. Joins state their expected cardinality. Labels remain null until a game is final. Missing external inputs remain null and receive coverage indicators; they are never silently converted to zero.

## Canonical entities

### `games`

Grain: one row per `game_id`.

Required: `game_id`, `season`, `week`, `season_type`, `start_date`, `home_team`, `away_team`, `neutral_site`, `conference_game`. Labels `home_score`, `away_score`, `home_win`, `home_margin`, and `total_points` are nullable until final. Scheduled games must not be dropped.

### `team_game_stats`

Grain: one row per (`game_id`, `team`).

Required: `game_id`, `season`, `team`, `home_away`. Raw box-score values are final-game observations and therefore become feature inputs only after a strict groupwise `shift(1)`. A duplicated key is a build failure.

### `line_snapshots`

Target grain: one row per (`game_id`, `sportsbook`, `market`, `side`, `captured_at`).

Fields: `line`, `odds`, `is_live`, `source`, `available_at`, plus an ingestion-run identifier. Markets are `moneyline`, `spread`, and `total`; sides are `home`, `away`, `over`, and `under`. Home spread follows sportsbook notation, so a seven-point home favorite is `-7`. Prices are American odds. Open/current/close are views selected from immutable snapshots, not columns whose meaning changes over time.

The current CFBD cache supplies only the latest provider values and cannot reconstruct historical line movement. `scripts/snapshot_market.py` now normalizes and appends forward-only CFBD observations to this schema, and a daily workflow collects them prospectively. CFBD can omit spread/total prices; a richer multi-book feed is still required for complete price-aware CLV.

The processed one-row-per-game `lines` consensus also retains `season`, median current/open spread and total, current-line dispersion, contributing book counts, and opening-to-current movement. Moneyline fair probability is calculated within provider before taking the cross-book median. The `season` partition is operationally important: refreshing 2026 replaces the entire prior 2026 consensus so withdrawn quotes cannot remain stale while historical seasons are preserved.

### `feature_observations`

Target grain: (`entity_id`, `feature_name`, `available_at`, `source_version`). This optional long-form store is the safest way to retain rankings, injuries, weather forecasts, roster news and market state. Training uses a backward as-of join from `prediction_time`.

### `feature_matrix`

Current grain: one row per `game_id` for a fixed prediction contract. Future multi-snapshot training should change the key to (`game_id`, `prediction_time`, `contract_name`). Required provenance fields are `prediction_time`, `feature_as_of`, `feature_set_version`, and coverage measures. The current hard contract prevents duplicate `game_id` rows.

`market_home_prob` is the normalized no-vig probability derived only when valid home and away moneylines are both present. `market_spread` uses sportsbook home-line notation; `market_total` is points. These mutable market fields are suitable for a current/fixed-snapshot prediction contract, but historical multi-snapshot training must instead resolve `line_snapshots` as of an explicit `prediction_time`.

### `model_backtest`

Grain: one row per `game_id` predicted in an expanding-season test fold. Fields include labels, market lines, `win_prob_oos`, `predicted_spread_oos`, and `predicted_total_oos`. A null OOS prediction means the row belonged to the initial training window or lacked an eligible label; it must not be filled with a final-model prediction.

### `bet_ledger`

Target grain: one row per placed bet. Fields include prediction and placement timestamps, game, market, side, selected line/odds/book, model probability, market probability, expected value, stake, closing line, CLV, result (`W`, `L`, `P`, `VOID`), profit, strategy version, model version and bankroll snapshot.

### `model_manifest`

Grain: one JSON document per model artifact version. It records code SHA, model version, training window, exact feature names, feature-schema hash, data fingerprint, parameters, OOS metrics, prediction contract and lifecycle status.

## Join map

```text
games (1/game)
  ├── team_game_stats (2/game) ──shift/roll by team──┐
  ├── line_snapshots (many/game) ──as-of/consensus───┤
  ├── feature observations (many/team or game) ─────┤
  └── chronological Elo (2 pregame ratings/game) ───┤
                                                    ▼
                              feature_matrix (1/game/contract)
                                                    │
                              season walk-forward models
                                                    ▼
                                model_backtest (1/game OOS)
                                                    │
                              price-aware selection/settlement
                                                    ▼
                                      bet_ledger (1/bet)
```

## Prediction contracts

- Win probability: `P(home wins)`, including declared neutral-site behavior.
- Spread: predicted `home_score - away_score`; positive means home by N.
- Sportsbook home spread: negative when the home team is favored.
- Home spread edge: `predicted_home_margin + home_spread`.
- Total: predicted `home_score + away_score`.
- Historical: only a saved walk-forward prediction may be displayed or graded.
- Future: a model trained through the latest completed game may score an unplayed game.

## Retention and versioning

- Raw: immutable source/year/capture partitions; never cache an empty error as success.
- Processed: reproducible from raw and versioned by schema.
- Processed refreshes: upsert refreshed season partitions or stable entity keys; never replace unrelated historical partitions merely because their raw cache was not loaded in a partial refresh.
- Features: reproducible from processed plus a feature-set version.
- Models: immutable artifacts plus manifests; aliases choose champion/challenger.
- Predictions and bets: append-only audit records.
