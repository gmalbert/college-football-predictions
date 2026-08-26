# Implementation priorities — August 2026

## Decision

The model foundation is complete enough to stop broad feature/model expansion. The priority is to restore a trustworthy operating cadence, collect prospective market evidence, and make releases reproducible. Win and spread remain market anchors; only the total-over model is eligible for prospective shadow validation.

## Verified current state

| Area | Status | Evidence | Implementation implication |
|---|---|---|---|
| Data/model contracts | Complete | Strict audit: 9 pass, 0 fail; 19,155 game and feature rows; no duplicate OOS rows | Preserve these invariants in every change |
| OOS model evaluation | Complete | v2.2 metrics and four manifests exist | Do not revive in-sample ATS reporting |
| Win and spread strategy | Hold | No win-market skill; spread equals market | No betting recommendations or complexity work |
| Total-over strategy | Shadow | Retrospective gate passed; prospective CLV absent | Collect evidence without changing model/threshold |
| Forward odds history | Started | 440 snapshots, initial 2026 schedule/line load | Insufficient for movement/CLV confirmation |
| Release operations | Needs repair | Best-bet and shadow exports are ~319 hours stale | Refresh before any user-facing recommendation |
| Context features | Adapter-ready | Safe transforms exist, inputs remain null | Add only timestamped sources |
| Automated unit tests | CI-ready; local execution blocked | 25 tests discovered; desktop runtime blocks SciPy DLL | Run in GitHub Actions or a normal Python environment |

## Priority 0 — restore safe daily operation

**Goal:** no stale output is visible or committed as current.

1. Run the runbook's refresh/rebuild/train/export/audit sequence in a normal project environment with `CFBD_API_KEY` configured.
2. Make the strict audit fail the release when either export exceeds its freshness window during the season; the audit currently reports those conditions only as warnings.
3. In both workflows, publish a single release metadata record with generation time, data fingerprint, model version, workflow run ID, and audit outcome.
4. Verify the workflows execute successfully in GitHub Actions; record the run URL/ID in release metadata rather than relying on local timestamps.

**Acceptance:** all current artifacts share one generation timestamp/fingerprint; strict audit has zero failures and no in-season freshness warnings; UI displays the release time and `hold`/`shadow` status.

## Priority 1 — collect prospective total-over evidence

**Goal:** produce a clean, append-only 2026 shadow ledger without model tuning.

1. Keep the daily and game-day snapshot workflow running; preserve raw provider payload and normalized rows with a capture/run ID.
2. Save each eligible total-over signal at its exact prediction time, including book, line, odds, model probability, market features, and model/strategy version.
3. After kickoff, select the final eligible pre-kickoff quote as close; settle the actual wager record (including pushes/voids) and calculate line and price CLV.
4. Lock the v2.2 feature set, regularization, and 57.5% threshold for the confirmation window. Any material change starts a new strategy version and resets the sample.
5. Publish prospective results by week, side, book, conference, market depth, and time-to-kickoff.

**Acceptance:** at least 150 predeclared eligible over signals or a full season; actual prices and closing quotes captured; positive mean CLV and documented confidence interval; no unversioned strategy changes.

## Priority 2 — harden market-data and release lineage

**Goal:** make every historic prediction and market comparison reconstructible.

1. Add immutable raw snapshot partitions by source/season/captured-at and an ingestion-run table.
2. Replace the generic processed `line_snapshots` append with a source-aware uniqueness key that preserves revisions and carries source response IDs when available.
3. Build a canonical `bet_ledger` from shadow signals rather than JSON-only exports.
4. Add release-level atomic publication: write artifacts to a staging directory, validate all of them, then promote the set together.
5. Retire or quarantine the 87 legacy empty cache files only after confirming their replacement/recovery path; do not delete blindly.

**Acceptance:** an arbitrary historical signal can be reconstructed from immutable raw market data, the model manifest, and release metadata; partial workflow failure cannot publish a mixed artifact set.

## Priority 3 — add only point-in-time contextual sources

**Goal:** expand information quality without leakage.

1. Start with one source family offering historical timestamps and clear licensing—QB/injury availability or issued weather forecasts are the most useful candidates.
2. Land each source as `feature_observations` with event time, `available_at`, source version, units, and revision semantics.
3. Add backward-as-of join and negative timing tests before a feature reaches an active allowlist.
4. Evaluate one family at a time against fixed walk-forward folds and the market baseline; retain only stable incremental value.

**Acceptance:** source coverage and timing are auditable; no current/final information can join to an earlier prediction; market-relative OOS results are reported before promotion.

## Priority 4 — gated research, not production changes

Only after Priorities 0–3 are healthy, run preregistered challenger experiments: early-season regime models, calibrated market residuals, CatBoost categorical effects, hierarchical pooling, recency decay, and likelihood-trained joint score models. Use frozen temporal folds, an explicit candidate log, price-aware ledgers, and a separate untouched confirmation period.

## Explicit non-priorities

- Do not optimize generic win/spread models until they beat the no-vig/implied-market baselines on identical OOS rows.
- Do not market retrospective total ROI as a live edge.
- Do not add un-timestamped injuries, weather, rankings, or season-final efficiency ratings to training.
- Do not relax release gates merely to make a dashboard show bets.

## Work sequence

```text
P0 fresh, coherent release
        ↓
P1 locked prospective total-over shadow ledger
        ↓
P2 immutable odds + release lineage
        ↓
P3 one timestamped contextual source at a time
        ↓
P4 preregistered challengers / promotion decision
```
