# 2026 schedule and market refresh

> **Implementation review — 2026-08-24:** **Completed for the initial load and repair.** The schedule, recovered historical provider lines, canonical snapshot artifact, partition-preserving upsert, and scheduled collector all exist. **Not yet complete operationally:** the current best-bet and shadow exports are stale (~319 hours on the review audit), only 440 forward observations exist, and current 2026 snapshots must continue through the season before they can support CLV or promotion.

Refresh timestamp: August 11, 2026 (America/New_York).

## Loaded artifacts

| Artifact | Result |
|---|---:|
| 2026 regular-season schedule | 1,638 unique games |
| 2026 postseason schedule | No rows returned yet; existing artifact left unchanged |
| 2026 CFBD line payload | 888 games |
| Current schedule rows with usable two-sided moneylines | 48 |
| Current schedule rows with a usable spread | 53 |
| Initial normalized line snapshot | 440 quotes |
| Full processed game table | 19,155 unique games |
| Recovered 2021–2025 line games | 6,936 API payloads with provider records |
| Historical opening spread coverage | 4,435 games |
| Historical opening total coverage | 4,463 games |
| Full feature matrix | 19,155 rows × 248 columns, zero duplicate game IDs |

The first scheduled kickoff in the current payload is August 27, 2026. Scores and training labels remain null for scheduled games. Models may score these rows for future display, but they cannot enter historical evaluation until final and assigned to a future walk-forward test fold.

## Partial-refresh incident and repair

The first schedule-only rebuild exposed a pre-existing ingestion flaw: processed builders assembled output only from raw cache files present in the selected rolling window. Because several older game caches were empty, a direct rebuild could temporarily replace the historical processed table with only 2026 rows. Historical games were immediately recovered from the validated one-row-per-game feature artifact and merged with the new schedule.

The permanent fix is `_save_processed_upsert` in `utils/fetch_historical.py`. Refreshed season partitions or stable keys now replace only matching processed rows; unrelated historical rows survive partial pulls. Empty upstream responses still do not overwrite non-empty caches. This invariant applies across games, lines, ratings, team-game stats, recruiting, and the optional processed feature sources.

## Market timing caveat

The 440-quote file is the start of prospective history, not a historical backfill. It can support movement and CLV only for predictions created at or after its capture time. The daily market workflow will append later snapshots; a true closing line is the last eligible pre-kickoff quote, never the mutable value currently returned by an API.

The archived CFBD endpoint did recover provider records with `spreadOpen` and `overUnderOpen` for 2021–2025. These support retrospective opening-to-final movement features, but they do not contain capture timestamps for every intermediate state. They therefore justify model development, not a claim of prospective execution. The 2026 append-only snapshots are the promotion evidence.
