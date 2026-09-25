# Streamlit vs FastAPI + React — measured differences

Generated 2026-09-20T21:42:15-0400 · 8 cold-context page loads per
page per stack · viewport 1440×900.

## 1. Content parity

| Dimension | Count |
|---|---:|
| Pages compared (content) | 10 |
| Pages matching exactly | **10** |
| Pages compared (live chrome) | 9 |
| Pages matching exactly (live) | **9** |
| Mismatched fields | **0** |
| Data tables compared | 16 |
| Table rows compared | 894 |
| **Table cells compared cell-by-cell** | **8,639** |
| Metrics compared | 58 |
| Captions compared | 19 |
| Alerts compared | 7 |
| Charts compared | 12 |

## 2. API latency (FastAPI only — Streamlit has no JSON API)

`cold` is the first request after explicitly clearing the artifact cache,
i.e. the cost of rebuilding every derived frame from Parquet. The server
also primes these caches on startup in a background thread, so a real
first request never pays it: measured end-to-end, the first Weekly
Predictions request after boot is ~160 ms rather than ~5.6 s.

| Page | Endpoint | Cold (ms) | Warm median (ms) | Warm stdev | Warm p95 (ms) | Payload (KB) |
|---|---|---:|---:|---:|---:|---:|
| Home | `/api/home` | 384 | 217.1 | 60.8 | 382.1 | 1.4 |
| Weekly Predictions | `/api/weekly-predictions` | 5594 | 96.0 | 5.7 | 108.9 | 104.4 |
| Value Bets | `/api/value-bets` | 360 | 356.2 | 17.1 | 398.3 | 8.3 |
| Team Explorer | `/api/team-explorer` | 421 | 271.9 | 16.5 | 300.2 | 35.0 |
| Historical Analysis | `/api/historical-analysis` | 311 | 252.3 | 34.2 | 349.2 | 38.8 |
| Model Performance | `/api/model-performance` | 140 | 147.1 | 19.1 | 193.9 | 19.1 |
| Win Probability | `/api/win-probability` | 837 | 61.1 | 3.7 | 64.5 | 55.4 |
| Preseason Outlook | `/api/preseason-outlook` | 237 | 218.9 | 18.5 | 236.8 | 57.4 |
| Data & Model Quality | `/api/data-quality` | 124 | 105.2 | 12.6 | 138.9 | 2.9 |

## 3. Browser timing (median of 8 loads)

`Heading` = navigation start → `<h1>` painted. `Settled` = additional time
after that until the page stops issuing requests. `p` is a two-sided
Mann-Whitney U test on the heading samples; Cliff's δ is the effect size
(+1 means every React sample beat every Streamlit sample).

| Page | Streamlit heading | React heading | Speed-up | p | Cliff's δ | Streamlit settled | React settled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Home | 337 ms | 142 ms | **2.38×** | 0.0002 | +1.00 | 831 ms | 767 ms |
| Weekly Predictions | 579 ms | 165 ms | **3.51×** | 0.0002 | +1.00 | 781 ms | 894 ms |
| Value Bets | 545 ms | 130 ms | **4.19×** | 0.0002 | +1.00 | 1198 ms | 1332 ms |
| Team Explorer | 491 ms | 123 ms | **3.99×** | 0.0002 | +1.00 | 1613 ms | 1912 ms |
| Historical Analysis | 537 ms | 146 ms | **3.69×** | 0.0002 | +1.00 | 1806 ms | 2080 ms |
| Model Performance | 570 ms | 140 ms | **4.08×** | 0.0002 | +1.00 | 1139 ms | 816 ms |
| Win Probability | 403 ms | 240 ms | **1.68×** | 0.0002 | +1.00 | 858 ms | 564 ms |
| Preseason Outlook | 547 ms | 140 ms | **3.91×** | 0.0002 | +1.00 | 1099 ms | 2042 ms |
| Data & Model Quality | 393 ms | 140 ms | **2.81×** | 0.0104 | +0.75 | 844 ms | 726 ms |

Aggregate: Streamlit **537 ms** vs React **140 ms** median (3.84× faster).

## 4. Transfer size

| Page | Streamlit (KB) | React (KB) | Δ | Streamlit requests | React requests |
|---|---:|---:|---:|---:|---:|
| Home | 2269 | 1657 | -613 | 103 | 8 |
| Weekly Predictions | 1428 | 1647 | +219 | 140 | 9 |
| Value Bets | 1796 | 1672 | -124 | 145 | 9 |
| Team Explorer | 5117 | 6291 | +1174 | 140 | 10 |
| Historical Analysis | 6214 | 6110 | -104 | 140 | 10 |
| Model Performance | 5582 | 1592 | -3990 | 124 | 9 |
| Win Probability | 5382 | 488 | -4894 | 146 | 10 |
| Preseason Outlook | 5988 | 5152 | -836 | 139 | 9 |
| Data & Model Quality | 1476 | 1576 | +100 | 125 | 8 |

## 5. Visual difference (matched screenshots, per-pixel)

A pixel counts as different when any RGB channel moves by more than 8. Both stacks rendered at the same viewport; the sidebar
strip is x < 300 px.

`Raw` is the difference with the two screenshots compared as-is. `Aligned`
re-renders the comparison at the vertical offset that minimises the
difference (searched ±90 px), which separates
*content in a different place* from *content that is different*.

| Page | Raw differing | Aligned differing | Best shift | Mean abs diff | Sidebar raw | Main raw |
|---|---:|---:|---:|---:|---:|---:|
| Home | 14.93% | **10.83%** | +40 px | 10.41 | 8.48% | 16.62% |
| Weekly Predictions | 30.99% | **26.97%** | +16 px | 11.55 | 21.13% | 33.59% |
| Value Bets | 31.05% | **22.83%** | +35 px | 12.62 | 21.13% | 33.66% |
| Team Explorer | 39.85% | **38.82%** | +4 px | 10.34 | 21.12% | 44.78% |
| Historical Analysis | 45.00% | **44.32%** | +3 px | 10.83 | 21.14% | 51.29% |
| Model Performance | 41.36% | **37.83%** | +16 px | 12.78 | 21.16% | 46.68% |
| Win Probability | 28.30% | **20.62%** | +16 px | 11.65 | 21.12% | 30.19% |
| Preseason Outlook | 37.08% | **36.51%** | -62 px | 17.23 | 21.41% | 41.20% |
| Data & Model Quality | 35.60% | **34.67%** | +6 px | 15.07 | 21.16% | 39.40% |

Aggregate raw difference **35.60%** of pixels;
after alignment **34.67%** (median best shift +16 px).

Alignment removes only 3% of the difference, so a single uniform vertical offset does **not** explain the residual: the spacing between elements differs by different amounts down the page, and the rendering chrome (canvas dataframes, Streamlit's toolbar, the Plotly theme) is genuinely different.

Interpretation: the visual metric measures *rendering*, not *content*. A 35%
pixel difference alongside 0 mismatched fields (8,639/8,639 table cells, all
headings, metrics, captions, alerts and navigation identical) means the two apps
put the same information on screen in slightly different boxes.

## 6. Where the remaining differences come from

- **`st.dataframe` is a canvas, not a table.** Streamlit paints every dataframe
  with glide-data-grid onto a `<canvas>`; the React build emits a real HTML
  `<table>`. Cell contents are verified identical cell-by-cell, but grid chrome
  (column headers, resize handles, per-cell toolbars, row striping offsets) is
  drawn differently. This is the single largest contributor on the table-heavy
  pages and explains why the main-content difference grows with page size
  (Home 17% → Historical Analysis 50%).
- **Streamlit chrome.** The `Deploy` button, hamburger menu and per-element
  toolbars exist only in Streamlit and sit in the top-right of every page.
- **Plotly theming.** Both render the same figure JSON, but Streamlit layers its
  own Plotly template over the traces, so axis lines and gridlines can differ.
- **Non-uniform vertical rhythm.** Streamlit spaces top-level elements with a
  flex gap plus per-element margins/padding; the React build approximates it.
  Element heights now match exactly (e.g. metric blocks are 76px in both, on a
  92px pitch) but the accumulated offset between sections still differs by
  tens of pixels.

Fonts are **not** a source of difference: the React build bundles the same
variable webfonts Streamlit ships (`SourceSansVF`, `SourceCodeVF`) and declares
them under the same family names, so both rasterise text identically.
