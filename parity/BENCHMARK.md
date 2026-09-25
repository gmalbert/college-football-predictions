# Parity build — speed and payload benchmark

Generated 2026-09-18T09:15:13-0400 · 5 iterations per measurement.

## 1. FastAPI endpoint latency

`cold` is the first request after clearing the artifact cache; `warm` is
the median of the repeated requests that follow.

| Page | Endpoint | Cold (ms) | Warm median (ms) | Warm p95 (ms) | Payload (KB) |
|---|---|---:|---:|---:|---:|
| Home | `/api/home` | 281.8 | 164.32 | 198.82 | 1.6 |
| Weekly Predictions | `/api/weekly-predictions` | 585.6 | 411.80 | 464.01 | 104.4 |
| Value Bets | `/api/value-bets` | 344.2 | 311.78 | 315.65 | 8.3 |
| Team Explorer | `/api/team-explorer` | 259.8 | 245.72 | 254.84 | 35.0 |
| Historical Analysis | `/api/historical-analysis` | 228.0 | 165.98 | 172.90 | 38.8 |
| Model Performance | `/api/model-performance` | 123.9 | 131.92 | 173.50 | 19.1 |
| Win Probability | `/api/win-probability` | 768.6 | 30.50 | 49.22 | 55.4 |
| Preseason Outlook | `/api/preseason-outlook` | 136.4 | 120.65 | 124.29 | 57.4 |
| Data & Model Quality | `/api/data-quality` | 85.7 | 79.09 | 79.65 | 2.9 |

## 2. Browser page load (median of 5 cold-context loads)

`Heading` is the time from navigation start until the page `<h1>` is
painted — the point at which the page is readable.

| Page | Streamlit heading (ms) | React heading (ms) | Speed-up | Streamlit settled (ms) | React settled (ms) | Streamlit KB | React KB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Home | 328 | 60 | 5.48× | 674 | 1 | 5050.3 | 6537.0 |
| Weekly Predictions | 527 | 259 | 2.04× | 1180 | 1 | 6569.4 | 5538.2 |
| Value Bets | 496 | 65 | 7.61× | 896 | 1 | 6583.3 | 5442.5 |
| Team Explorer | 472 | 151 | 3.13× | 1412 | 1 | 11104.8 | 6570.4 |
| Historical Analysis | 443 | 171 | 2.59× | 1149 | 1 | 11104.8 | 6574.2 |
| Model Performance | 387 | 39 | 9.90× | 1026 | 1 | 10984.4 | 5453.3 |
| Win Probability | 401 | 35 | 11.36× | 727 | 0 | 10104.1 | 5545.0 |
| Preseason Outlook | 426 | 173 | 2.46× | 835 | 1 | 10003.6 | 5491.6 |
| Data & Model Quality | 480 | 48 | 10.02× | 575 | 1 | 6687.3 | 6538.3 |

## 3. Summary

- Median page heading-visible time — Streamlit **443 ms** vs React **65 ms** (6.81× faster).
- Warm API median across all endpoints — **164.32 ms**.
- Largest JSON payload — **104.4 KB** (Weekly Predictions).
