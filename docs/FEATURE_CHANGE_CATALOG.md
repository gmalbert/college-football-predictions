# Feature and change catalog

This catalog contains 85 concrete additions. “Implemented” means executable code is in this repository. “Adapter-ready” means the transform, contract and missing-data behavior exist, but an external timestamped feed must populate the named inputs. That boundary is important: no code can recreate historical injury, weather-forecast or sportsbook snapshots that were never collected.

> **Implementation review — 2026-08-24:** Items 1–85 are **implemented in the repository** and their central artifacts are present. The strict audit verifies the primary contracts, OOS output, manifests, and snapshot artifact with zero failures. This does **not** mean every capability is production-complete: items 44–48 and 61/83/84 depend on continued prospective quote collection; items 50–54 are risk primitives rather than proof of a live edge; and every contextual family below remains adapter-ready until a timestamped source is connected.

| # | Change | Status | Code / model impact |
|---:|---|---|---|
| 1 | One-game feature grain contract | Implemented | `utils.contracts.FEATURE_CONTRACT` blocks duplicate game rows |
| 2 | Team-game grain contract | Implemented | Enforces (`game_id`, `team`) uniqueness |
| 3 | Guarded merge cardinality | Implemented | `safe_merge` validates many-to-one/one-to-one and row growth |
| 4 | Point-in-time availability assertion | Implemented | Rejects features observed after prediction time |
| 5 | Backward as-of feature join | Implemented | `point_in_time_join` selects only prior observations |
| 6 | UTC normalization | Implemented | Shared timezone-aware timestamp parsing |
| 7 | Duplicate team-stat repair | Implemented | Dedup in ingestion and feature construction |
| 8 | Game-key rolling-stat join | Implemented | Eliminates the prior team/week many-to-many explosion |
| 9 | Empty-cache rejection | Implemented | Empty JSON cannot become a permanent successful cache |
| 10 | Dynamic season window | Implemented | Five completed seasons plus current year; optional CLI years |
| 11 | Live-season refresh | Implemented | Current season refreshes on normal scheduled runs |
| 12 | Scheduled-game retention | Implemented | Null labels remain in `games` for future inference |
| 13 | Chronological pregame Elo | Implemented | Updates only after completed games; preseason mean reversion |
| 14 | Margin-of-victory Elo adjustment | Implemented | Larger results move ratings more, with bounded multiplier |
| 15 | Neutral-site home advantage | Implemented | Pregame Elo respects neutral venues |
| 16 | Cross-role rest calculation | Implemented | Previous game found whether team was home or away |
| 17 | Shifted rolling team stats | Implemented | Current game excluded before 5-game rolling averages |
| 18 | Shifted rolling scoring/margin | Implemented | Points for/against and margin derived on team-long history |
| 19 | Rolling rushing and passing yards | Implemented | Matchup form without current-game leakage |
| 20 | Rolling turnovers | Implemented | Home/away and differential form features |
| 21 | Rolling discipline | Implemented | Penalty-yard differential when source coverage exists |
| 22 | Rolling third-down efficiency | Implemented | Team and matchup values |
| 23 | Rolling red-zone efficiency | Implemented | Safe ratio from shifted attempts/conversions |
| 24 | Rolling sacks and total yards | Implemented | Team and differential form features |
| 25 | Safe active feature allowlist | Implemented | Season-final SP+/FPI/SRS/EPA/PPA/WEPA excluded from models |
| 26 | Explicit missing handling | Implemented | XGBoost native missing paths; median+indicator for sklearn |
| 27 | Context coverage indicators | Implemented | Missing optional sources are measurable, not hidden |
| 28 | Expanding-season walk-forward validation | Implemented | Earlier seasons train; next season is the test fold |
| 29 | Persisted OOS prediction artifact | Implemented | Honest calibration, ATS and historical UI source |
| 30 | Market probability baseline | Implemented | Devigged home/away moneyline probability |
| 31 | Market margin and total baselines | Implemented | OOS RMSE/MAE compare model with available line |
| 32 | Brier/log-loss/accuracy metrics | Implemented | Probability quality measured OOS |
| 33 | Expected calibration error | Implemented | Reliability gap weighted across bins |
| 34 | Calibration table/curve | Implemented | Saved OOS predictions power UI diagnostics |
| 35 | Split-conformal intervals | Implemented | Reusable interval primitive and coverage/width scoring |
| 36 | Joint margin/total score distribution | Implemented | Coherent win, cover, over probabilities and score means |
| 37 | Residual correlation estimation | Implemented | Fits margin/total uncertainty and dependence |
| 38 | Temporal-cluster bootstrap intervals | Implemented | Resamples season-week clusters rather than iid bets |
| 39 | Correct spread sign convention | Implemented | `edge = home margin forecast + home spread` everywhere |
| 40 | Exact spread/total/moneyline settlement | Implemented | Wins, losses, pushes and voids with price-aware profit |
| 41 | American/decimal odds conversion | Implemented | Shared validated price math |
| 42 | Vig removal and overround | Implemented | Fair market probability baseline |
| 43 | Expected value by price | Implemented | Selection can require positive EV, not only point edge |
| 44 | Quote as-of selection | Implemented | Never reads a quote after prediction time |
| 45 | Multi-book consensus | Implemented | Median line, price and book count |
| 46 | Line movement, dispersion and staleness | Implemented | Market-quality features from snapshot frames |
| 47 | Closing-line value | Implemented | Price/line improvement measured consistently by market |
| 48 | OOS spread/total bet ledgers | Implemented | Reproducible selection and settlement with thresholds |
| 49 | Strict OOS ledger assertion | Implemented | Training cutoff must precede prediction time |
| 50 | Probability shrinkage to market | Implemented | Reduces fragile model-only edges |
| 51 | Standard-error uncertainty haircut | Implemented | Prevents full staking on noisy estimates |
| 52 | Fractional Kelly sizing | Implemented | Growth logic tempered by estimation risk |
| 53 | Bet/game/team/slate caps | Implemented | Hard bankroll concentration controls |
| 54 | Correlation penalty | Implemented | Related positions receive smaller allocations |
| 55 | Data and schema fingerprints | Implemented | Reproducible artifact identity |
| 56 | Model manifests | Implemented | Features, window, metrics, parameters and contracts per model |
| 57 | Drift monitoring (PSI + null shift) | Implemented | Feature-level watch/alert report |
| 58 | Champion/challenger promotion gates | Implemented | Holds release when quality or CLV requirements fail |
| 59 | Data-quality dashboard and strict audit CLI | Implemented | Operators see contract, freshness and leakage risks |
| 60 | CI tests and release audit | Implemented | Pipeline tests before build and audits before commit |
| 61 | Forward-only market snapshot collector | Implemented | Canonical CFBD quote normalization, atomic append and daily scheduled capture |
| 62 | Partition-preserving processed upserts | Implemented | A current-season refresh cannot erase absent historical raw partitions |
| 63 | Derived no-vig market feature | Implemented | `market_home_prob` is materialized from valid two-sided prices |
| 64 | Market-anchored win inference | Implemented | Uses no-vig price when available and a structural classifier otherwise |
| 65 | Market-anchored margin inference | Implemented | Uses implied market margin when available and XGBoost otherwise |
| 66 | Nested residual shrinkage | Implemented | Correction magnitude is learned only from inner chronological OOS predictions |
| 67 | Market-residual total model | Implemented | Ridge predicts total-market error and applies OOS-estimated shrinkage |
| 68 | Multi-family challenger benchmark | Implemented | Calibration, structural logistic, Ridge, histogram boosting and ExtraTrees candidates |
| 69 | Actionable-edge abstention | Implemented | Spread grading ignores market-parity rows below a one-point edge |
| 70 | 2026 schedule integration | Implemented | 1,638 scheduled games retained with null labels for future inference |
| 71 | Initial 2026 odds snapshot | Implemented | 888 game-line payloads pulled; 440 normalized quotes captured prospectively |
| 72 | Historical provider-line recovery | Implemented | Re-pulled 2021–2025 raw line payloads after detecting empty caches |
| 73 | Robust line consensus | Implemented | Median current/open spread and total replace arithmetic means |
| 74 | Per-book moneyline de-vigging | Implemented | Fair probability is computed within provider before cross-book median |
| 75 | Opening spread/total fields | Implemented | Preserves CFBD `spreadOpen` and `overUnderOpen` |
| 76 | Opening-to-current movement | Implemented | Separate spread and total move features |
| 77 | Cross-book dispersion | Implemented | Standard deviation identifies disagreement and unstable markets |
| 78 | Market depth | Implemented | Book counts distinguish one-book lines from broader consensus |
| 79 | Dedicated total-side classifier | Implemented | Regularized logistic model predicts over versus under directly |
| 80 | Closing-time prediction contract | Implemented | Signals eligible only inside a 12-hour pre-kickoff window |
| 81 | Fixed probability-edge selection | Implemented | Requires selected-side probability of at least 57.5% |
| 82 | Side-specific promotion | Implemented | Overs pass retrospective stability; unders remain held |
| 83 | Prospective shadow ledger | Implemented | Separate JSON contains non-actionable 2026 validation signals |
| 84 | Game-day market refresh | Implemented | Additional scheduled snapshots rebuild market features and shadow output |
| 85 | Total Market Signals dashboard | Implemented | Shows OOS record, confirmation season, gates, and current shadow state |

## Adapter-ready contextual feature families

`utils.advanced_features.build_context_features` implements more than 50 additional transforms with safe null behavior. Connectors must supply historical, timestamped inputs before these enter a model.

| Family | Implemented transforms | Required source work |
|---|---|---|
| Calendar | season progress, early/late/postseason, neutral, conference/cross-conference | Already available from schedule |
| Rankings | poll differential, ranked matchup, ranked-vs-unranked, trap spot | Weekly poll snapshots with release timestamps |
| Schedule | short week, bye, rest advantage | Already derivable; next-opponent rank needs poll join |
| Travel | distance differential, timezone shift, body-clock penalty | Team/venue coordinates and local kickoff time |
| Venue | altitude acclimation, capacity pressure, dome, surface | Typed venue history, including venue changes |
| Weather | freeze, heat, wind, precipitation, composite, pass-style interaction | Archived forecast issued before prediction—not observed weather |
| Roster | QB availability/uncertainty, roster continuity, coordinator continuity | Timestamped depth chart, injury and staff news feed |
| Talent | returning production, portal net, recruiting points, coach tenure | Preseason/versioned snapshots |
| Efficiency | turnover regression, explosive balance, pace, possessions, pressure, penalties, red zone, sacks, special teams | Shifted play/drive/team-game aggregates |
| Game type | rivalry, bowl, rematch, FCS mismatch | Curated rivalry/game metadata |
| Market quality | move, disagreement, depth, quote staleness | Immutable multi-book quote snapshots |

## Model experiments after data collection

These are deliberately gated behind clean data rather than baked into the first refactor: CatBoost with categorical team/conference effects, hierarchical partial pooling, monotonic constraints around market priors, an Elo-only baseline, market-residual targets, separate early-season models, recency-decay ensembles, calibration by season regime, and a joint score model trained directly by likelihood. Each must use the same walk-forward folds and be compared to market baselines with a locked evaluation plan.
