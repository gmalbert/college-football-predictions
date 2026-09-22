# FastAPI + React parity build

A second front end for Tailgate Edge that renders the **same site** as the
Streamlit app, backed by FastAPI instead of Streamlit's script runtime.

The Streamlit application is unchanged and remains the source of truth. The
React build exists to prove the dashboard can run as a conventional
API + SPA with no loss of content.

```
browser ──▶ React (Vite build, served by FastAPI)  ──▶  /api/*  ──▶  utils/  ──▶  data_files/
browser ──▶ Streamlit (unchanged)                    ──▶           utils/  ──▶  data_files/
```

Both front ends call the **same `utils/` functions**, so every number, string
and recommendation is produced by identical code. The API layer contains no
modelling or formatting logic of its own beyond turning the pages' display
strings into JSON.

---

## Layout

| Path | Purpose |
|---|---|
| `api/main.py` | FastAPI app, navigation definition, timing middleware, static SPA mount |
| `api/services/` | One module per Streamlit page; each returns that page's full payload |
| `api/charts.py` | Serialises Plotly figures with Streamlit's colourway |
| `api/jsonutil.py` | pandas/NumPy → JSON, plus the pages' display formatters |
| `api/data.py` | mtime-keyed artifact cache (the `@st.cache_data` equivalent) |
| `frontend/src/components/` | Streamlit-compatible primitives (`stMetric`, `stCaption`, `stAlert`, `stDataFrame`, `stTabs`, `stExpander`, …) |
| `frontend/src/pages/` | One component per Streamlit page |
| `frontend/src/styles.css` | The Frost theme, taken from `utils/ui_components.py` |
| `scripts/parity_snapshot.py` | Extracts ground truth from Streamlit via `AppTest` |
| `scripts/parity_check.py` | Playwright comparison of React against that ground truth |
| `scripts/benchmark.py` | Speed and payload benchmark |
| `scripts/verify_parity.py` | Runs the whole verification pipeline |
| `parity/expected/` | Committed ground-truth snapshots, one per page |
| `parity/parity_report.json` | Machine-readable result of the last parity run |
| `parity/BENCHMARK.md` | Last benchmark report |
| `api/settings.py` | Every `TAILGATE_*` environment variable, parsed and documented |
| `api/__main__.py` | `python -m api` — runs uvicorn from those settings |

---

## Running it

```bash
# one-time
python -m pip install -r requirements-web.txt
python -m playwright install chromium
cd frontend && npm install && npm run build && cd ..

# terminal 1 — Streamlit (unchanged)
python -m streamlit run predictions.py --server.port 8501

# terminal 2 — FastAPI + the built React app
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# → http://127.0.0.1:8000
```

For front-end development, `npm run dev` in `frontend/` serves Vite on
`:5173` and proxies `/api` to `:8000`.

---

## Configuration

All runtime configuration comes from the environment, so nothing needs editing
to deploy. `python -m api` reads the same variables and passes them to uvicorn.

| Variable | Default | Effect |
|---|---|---|
| `TAILGATE_HOST` | `127.0.0.1` | Bind address |
| `TAILGATE_PORT` | `8000` | Bind port |
| `TAILGATE_WORKERS` | `1` | Uvicorn worker processes |
| `TAILGATE_LOG_LEVEL` | `warning` | Uvicorn log level |
| `TAILGATE_CORS_ORIGINS` | *(empty)* | Comma-separated browser origins allowed to call the API cross-origin |
| `TAILGATE_API_KEY` | *(empty)* | When set, every `/api/*` request except `/api/health` needs `X-API-Key` or `Authorization: Bearer` |
| `TAILGATE_ADMIN_TOKEN` | *(empty)* | Enables `POST /api/cache/clear`; when empty that endpoint returns 404 |
| `TAILGATE_WARM_CACHE` | `true` | Prime the artifact caches at startup |
| `TAILGATE_LOGO_WIDTH` | `400` | Width `/api/logo` is downscaled to |

### The auth decision

The API is **read-only and public by default**, matching the Streamlit app it
replaces — anyone who can reach it can read the same public predictions. Two
things are locked down regardless:

- **`POST /api/cache/clear` is admin-only.** It forces a multi-second rebuild
  from Parquet, so leaving it open was a trivial denial-of-service. It now
  returns 404 unless `TAILGATE_ADMIN_TOKEN` is set, and then requires that
  token.
- **CORS is closed by default.** FastAPI serves the SPA itself, so browser
  requests are same-origin and need no CORS headers. The middleware is only
  mounted when `TAILGATE_CORS_ORIGINS` names specific origins — the previous
  `allow_origins=["*"]` was pure surface area.

`TAILGATE_API_KEY` is available for deployment shapes that expose the JSON API
to something other than the bundled SPA. It is deliberately **not** the primary
control: the SPA fetches from the browser and cannot carry a secret, so if the
whole site needs protecting, terminate auth at the reverse proxy.

### Workers and the cache

`TAILGATE_WORKERS=N` spawns N processes, and the artifact cache lives in
process memory — so N workers means N copies of it and N warm-ups. The server
prints a warning at startup when that combination is configured; set
`TAILGATE_WARM_CACHE=false` to let each worker populate its cache on demand if
the memory is not worth it.

### Verified behaviour

```
open instance   (no TAILGATE_* set)
  GET  /api/health              200
  GET  /api/home                200
  POST /api/cache/clear         404   (disabled)

secured instance (API key + admin token)
  GET  /api/health              200   (intentionally open for healthchecks)
  GET  /api/home  no key        401
  GET  /api/home  wrong key     401
  GET  /api/home  X-API-Key     200
  GET  /api/home  Bearer        200
  POST /api/cache/clear  key    403
  POST /api/cache/clear  admin  200
```

---

## How parity is verified

Two independent checks run on every verification pass, because no single
technique can see the whole picture.

### 1. Content parity — `scripts/parity_check.py --content`

`streamlit.testing.v1.AppTest` executes each page in-process and exposes the
resulting element tree, **including the complete contents of every
`st.dataframe`** — data that the browser paints onto a canvas and therefore
cannot be scraped from the live DOM.

`parity_snapshot.py` writes that tree to `parity/expected/<page>.json`.
Playwright then drives the React app and extracts the same signature from its
DOM. The two are compared field by field:

- every heading (`h1`/`h2`/`h3`) and caption, in order
- every metric — label, value **and** delta
- every alert with its severity
- every markdown block
- every expander label
- **every table, cell by cell** (all rows, all columns)

### 2. Live chrome parity — `scripts/parity_check.py --live`

The live Streamlit server and the live React server are loaded in the same
browser and compared on the things AppTest cannot observe:

- sidebar navigation — link text, order, and which link is marked current
- widget labels
- Plotly chart count and chart titles
- expander labels
- table count
- footer presence

### 3. Behaviour — `tests/test_web_e2e.py`

51 Playwright tests drive the real UI: changing the season, week, conference,
minimum-edge slider and bet type; switching every tab; toggling expanders;
typing in the game search; changing the sidebar season filter. They also
assert there are no console errors on any route.

### 4. Contracts — `tests/test_api_parity.py`

101 pytest cases covering each endpoint's payload shape, the navigation
definition, the timezone-dependent kickoff column, and a byte-compilation
sweep over all 78 Python modules in the repository.

### Result

```
home                   content=pass  live=pass
Weekly_Predictions     content=pass  live=pass
Value_Bets             content=pass  live=pass
Team_Explorer          content=pass  live=pass
Historical_Analysis    content=pass  live=pass
Model_Performance      content=pass  live=pass
Win_Probability        content=pass  live=pass
Preseason_Outlook      content=pass  live=pass
Data_Quality           content=pass  live=pass
Total_Market_Signals   content=pass  live=n/a
failures: 0
```

---

## Intentional differences

Everything below is deliberate and documented in the code.

**`Total_Market_Signals` is unlinked, not absent.** `pages/10_Total_Market_Signals.py`
is not registered in `predictions.py::nav_sections`, so the live Streamlit
server answers **404** for it (tracked as U1 in `docs/UI_UX_ENHANCEMENTS.md`).
The React build keeps the route reachable by direct URL but — like Streamlit —
does not list it in the sidebar. The live check asserts the Streamlit 404
rather than comparing against an error page.

**Two pages have no footer.** `pages/7_Win_Probability.py` and
`pages/8_Preseason_Outlook.py` never call `add_betting_oracle_footer()`. The
React build reproduces both omissions.

**Streamlit rendering artefacts are normalised, not copied.** AppTest reports
raw Markdown source and un-rendered HTML, while the browser DOM reports painted
text. The comparator strips emphasis markers, inline-code backticks, HTML tags
and the Material-icon ligature that Streamlit puts in expander summaries. It
also tolerates two things that genuinely change between two runs minutes apart:
wall-clock-derived freshness ages, and the 0.1% drift in floating-point
rounding of those ages.

**Alert icons.** Streamlit draws its alert icon as a Material font glyph, which
contributes nothing to the text layer; React uses an emoji, which does. The
comparator strips the leading icon before comparing.

---

## Performance

Full numbers in [`parity/BENCHMARK.md`](parity/BENCHMARK.md). Headlines from a
5-iteration run on this machine:

| Metric | Streamlit | FastAPI + React |
|---|---:|---:|
| Median time to page heading (9 pages) | 443 ms | **65 ms** (6.8× faster) |
| Median warm API response | — | 164 ms |
| Largest JSON payload | — | 104 KB (Weekly Predictions) |
| Per-page browser transfer | 5.0–11.1 MB | 5.4–6.6 MB |

The React bundle is dominated by `plotly.js` (~4.8 MB raw, ~1.4 MB gzipped);
after the first page load it is served from the HTTP cache, which is why the
warm page transitions are effectively instant. Streamlit's transfer grows with
the amount of data on a page because the runtime ships the full dataframe
arrow payload to the client on every re-run.

Two API endpoints are worth noting:

- `GET /api/win-probability` is 769 ms cold but **30 ms warm** — the cold
  call is a live CFBD request; the result is memoised for the process lifetime.
- `GET /api/weekly-predictions` is the heaviest endpoint (104 KB) because the
  compact table ships all 300+ games for the selected week as pre-formatted
  strings, exactly as the Streamlit page renders them.

---

## Commands

```bash
# everything
python scripts/verify_parity.py

# individual steps
python scripts/parity_snapshot.py                      # refresh ground truth
python scripts/parity_check.py --content --live --json # compare
python -m pytest tests/test_api_parity.py -q           # API + py_compile
python -m pytest tests/test_web_e2e.py -q              # Playwright E2E
python scripts/benchmark.py --iterations 5             # speed report
python scripts/stats_diff.py --iterations 8            # full statistical report
python scripts/profile_weekly.py                       # where a page's time goes
```

### Ports

The harness expects Streamlit on `:8501` and the API on `:8000`. Both are
overridable, which matters because `8501` is Streamlit's default and another
project's app can already be sitting on it:

```bash
PARITY_STREAMLIT_BASE=http://127.0.0.1:8511 \
PARITY_REACT_BASE=http://127.0.0.1:8000 \
python scripts/parity_check.py --live
```

Before comparing anything, the harness loads both roots and refuses to run if
the page does not look like Tailgate Edge. This exists because a foreign
Streamlit app on `:8501` once produced a wall of failures that looked like
regressions but were simply a different application.

---

## Not yet done

The application is feature-complete and verified; the packaging is not.

- **Not committed.** `api/`, `frontend/`, `parity/`, `scripts/` and the two new
  test files are all still untracked.
- **No Dockerfile or CI workflow.** `frontend/dist/` is gitignored, so a fresh
  clone serves a JSON notice at `/` until `npm ci && npm run build` runs. The
  API itself works immediately — verified by
  `scripts/check_no_build.py`.
- **No reverse proxy.** TLS, compression and (if wanted) site-wide
  authentication belong in front of the API, not in it.
- **No committed branch or PR.**

