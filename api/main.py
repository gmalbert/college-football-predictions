"""FastAPI application exposing the Streamlit dashboard's data as a JSON API.

The Streamlit app is the source of truth.  Every service module in
``api.services`` imports the *same* ``utils/`` helpers the pages use, so the
numbers, formatting and recommendation strings are produced by identical code.
The React client is a thin renderer over these payloads.
"""
from __future__ import annotations

import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from api import __version__
from api.data import DATA_DIR, artifact, cache_clear, cache_stats
from api.settings import settings
from api.services import (
    data_quality,
    historical,
    home,
    model_performance,
    preseason,
    team_explorer,
    total_signals,
    value_bets,
    weekly,
    win_probability,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGO_PATH = PROJECT_ROOT / "data_files" / "logo.png"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def _warm_caches() -> None:
    """Populate the expensive caches before the first request arrives.

    The first Weekly Predictions request of a process has to load the feature
    matrix, deserialise the XGBoost/Ridge models and build the season-wide
    market consensus — several seconds of work that every later request reuses
    from cache.  Doing it at startup keeps that cost off the first visitor.
    """
    started = time.perf_counter()
    try:
        weekly.build_weekly(timezone_name="UTC")
        home.build_home()
        meta()
    except Exception as exc:  # noqa: BLE001 - a warm-up failure must not block boot
        print(f"[warmup] skipped: {type(exc).__name__}: {exc}")
        return
    print(f"[warmup] caches primed in {(time.perf_counter() - started):.2f}s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import threading

    for note in settings.warnings():
        print(f"[config] {note}")
    print(f"[config] {settings.describe()}")

    if settings.warm_cache:
        # Run in a thread so the server starts accepting connections immediately.
        threading.Thread(target=_warm_caches, daemon=True).start()
    yield


app = FastAPI(
    title="Tailgate Edge API",
    version=__version__,
    description="Read-only JSON API backing the React parity build of the Streamlit dashboard.",
    lifespan=lifespan,
)

# CORS is only mounted when an explicit origin list is configured. FastAPI
# serves the SPA itself, so browser requests are same-origin and need no CORS
# headers at all; allowing "*" would only widen the surface for no benefit.
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Authorization", "Content-Type"],
    )

if settings.auth_enabled:

    @app.middleware("http")
    async def _require_api_key(request, call_next):
        """Require an API key on /api/* when TAILGATE_API_KEY is set.

        ``/api/health`` stays open so container healthchecks and load balancers
        can reach it without holding a credential. Static SPA assets are served
        outside /api and are likewise unaffected.
        """
        if request.url.path.startswith("/api/") and request.url.path != "/api/health":
            supplied = request.headers.get("x-api-key") or ""
            if not supplied:
                authorization = request.headers.get("authorization", "")
                if authorization.lower().startswith("bearer "):
                    supplied = authorization[7:].strip()
            if not secrets.compare_digest(supplied, settings.api_key or ""):
                return JSONResponse(
                    {"detail": "Unauthorized"}, status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

# ---------------------------------------------------------------------------
# Navigation — byte-for-byte the same sections/titles/icons as
# predictions.py::nav_sections.  ``/total-market-signals`` is deliberately
# absent: pages/10 is not registered in st.navigation, so it is not reachable
# from the Streamlit sidebar either.
# ---------------------------------------------------------------------------
NAV_SECTIONS = [
    {
        "section": "",
        "pages": [
            {
                "title": "Home",
                "icon": "🏈",
                "path": "/",
                "default": True,
                "page_title": "🏈 College Football Predictions",
            },
        ],
    },
    {
        "section": "Analysis",
        "pages": [
            {"title": "Weekly Predictions", "icon": "📊", "path": "/Weekly_Predictions",
             "page_title": "📊 Weekly Predictions"},
            {"title": "Value Bets", "icon": "💰", "path": "/Value_Bets",
             "page_title": "💰 Value Bets"},
            {"title": "Team Explorer", "icon": "🏟️", "path": "/Team_Explorer",
             "page_title": "🏟️ Team Explorer"},
            {"title": "Historical Analysis", "icon": "📈", "path": "/Historical_Analysis",
             "page_title": "📈 Historical Analysis"},
            {"title": "Model Performance", "icon": "🎯", "path": "/Model_Performance",
             "page_title": "🎯 Model Performance"},
            {"title": "Win Probability", "icon": "📉", "path": "/Win_Probability",
             "page_title": "📈 In-Game Win Probability"},
            {"title": "Preseason Outlook", "icon": "🔮", "path": "/Preseason_Outlook",
             "page_title": "🏈 Preseason Outlook"},
            {"title": "Data & Model Quality", "icon": "🛡️", "path": "/Data_Quality",
             "page_title": "🛡️ Data & Model Quality"},
        ],
    },
]

# Present in the repository but unlinked from st.navigation (docs/UI_UX_ENHANCEMENTS.md U1).
EXTRA_ROUTES = [
    {"title": "Total Market Signals", "icon": "📈", "path": "/Total_Market_Signals",
     "page_title": "Total Market Signals"},
]

APP_TITLE = "Tailgate Edge - College Football Predictions"
APP_ICON = "🏈"

_TIMINGS: dict[str, list[float]] = {}


@app.middleware("http")
async def _record_timing(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    if request.url.path.startswith("/api/"):
        bucket = _TIMINGS.setdefault(request.url.path, [])
        bucket.append(elapsed)
        del bucket[:-200]
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.2f}"
    return response


def _sidebar_metrics() -> dict:
    """Mirror ``utils.ui_components.render_sidebar``'s live metric block."""
    try:
        from utils.model_artifacts import load_metrics, models_trained

        if not models_trained():
            return {"metrics": []}
        metrics = load_metrics()
        ats = metrics.get("ats", {})
        win = metrics.get("win_model", {})
        cards = []
        if ats:
            cards.append(
                {
                    "label": "Season ATS",
                    "value": f"{ats.get('wins', 0)}‑{ats.get('losses', 0)}",
                    "delta": f"{ats.get('pct', 0):.1%} win rate",
                    "help": None,
                }
            )
        if win.get("brier"):
            cards.append(
                {
                    "label": "Model Brier",
                    "value": f"{win['brier']:.4f}",
                    "delta": None,
                    "help": "Lower is better",
                }
            )
        return {"metrics": cards}
    except Exception:  # noqa: BLE001 - parity with the Streamlit try/except
        return {"metrics": []}


@app.get("/api/health")
def health() -> dict:
    """Liveness probe. Deliberately unauthenticated and cheap."""
    return {
        "status": "ok",
        "version": __version__,
        "cache": cache_stats(),
        "config": settings.describe(),
    }


@app.get("/api/meta")
def meta() -> dict:
    """Global chrome: navigation, sidebar metrics, branding, cache stats."""
    return {
        "app_title": APP_TITLE,
        "app_icon": APP_ICON,
        "nav": NAV_SECTIONS,
        "extra_routes": EXTRA_ROUTES,
        "sidebar": {
            "logo": "/api/logo" if LOGO_PATH.exists() else None,
            **_sidebar_metrics(),
        },
        "footer": {
            "html": (
                "Powered by Betting Oracle\n"
                "Sports Prediction Analytics"
            ),
            "url": "https://www.betting-oracle.com",
            "logo": (
                "https://raw.githubusercontent.com/gmalbert/betting-oracle/"
                "main/data_files/logo.png"
            ),
        },
        "theme": {
            "primaryColor": "#2B7CB8",
            "backgroundColor": "#EEF4FB",
            "secondaryBackgroundColor": "#F7FBFF",
            "textColor": "#1A2B3C",
            "font": "sans serif",
        },
        "cache": cache_stats(),
    }


@app.get("/api/logo")
def logo():
    """Serve a downscaled copy of ``data_files/logo.png``.

    The source file is ~663 KB and is displayed at 120-200 px wide, so shipping
    it unscaled made the logo one of the larger transfers on every page load.
    The resized bytes are memoised against the source file's mtime.
    """
    if not LOGO_PATH.exists():
        raise HTTPException(status_code=404, detail="logo.png not found")

    def _render() -> bytes:
        from io import BytesIO

        from PIL import Image

        with Image.open(LOGO_PATH) as source:
            image = source.convert("RGBA")
            if image.width > settings.logo_width:
                ratio = settings.logo_width / image.width
                image = image.resize(
                    (settings.logo_width, max(1, round(image.height * ratio))),
                    Image.LANCZOS,
                )
            buffer = BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()

    payload = artifact(f"logo:resized:{settings.logo_width}", LOGO_PATH, _render)
    return Response(
        content=payload,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/cache/clear", include_in_schema=settings.admin_enabled)
def clear_cache_endpoint(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    """Drop the artifact cache so the next request rebuilds from Parquet.

    This forces a multi-second rebuild, so it is not left public: the endpoint
    only exists when ``TAILGATE_ADMIN_TOKEN`` is configured, and then requires
    that token.
    """
    if not settings.admin_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    if not x_admin_token or not secrets.compare_digest(
        x_admin_token, settings.admin_token or ""
    ):
        raise HTTPException(status_code=403, detail="Invalid admin token")
    cache_clear()
    return {"status": "cleared"}


@app.get("/api/stats")
def stats() -> dict:
    """Per-endpoint latency statistics collected by the timing middleware."""
    out = {}
    for path, samples in sorted(_TIMINGS.items()):
        if not samples:
            continue
        ordered = sorted(samples)
        out[path] = {
            "count": len(samples),
            "min_ms": round(ordered[0], 3),
            "median_ms": round(ordered[len(ordered) // 2], 3),
            "max_ms": round(ordered[-1], 3),
            "mean_ms": round(sum(ordered) / len(ordered), 3),
        }
    return {"endpoints": out, "cache": cache_stats()}


# ---------------------------------------------------------------------------
# Page endpoints
# ---------------------------------------------------------------------------


@app.get("/api/home")
def api_home() -> dict:
    return home.build_home()


@app.get("/api/weekly-predictions")
def api_weekly(
    season: int | None = None,
    week: int | None = None,
    conference: str = "All",
    min_edge: float = 0.0,
    sort_by: str = "Edge (High→Low)",
    tz: str | None = Query(default=None, description="IANA browser timezone"),
) -> dict:
    return weekly.build_weekly(
        season=season, week=week, conference=conference,
        min_edge=min_edge, sort_by=sort_by, timezone_name=tz,
    )


@app.get("/api/value-bets")
def api_value_bets(
    season: int | None = None,
    bet_type: str = "Spread",
    min_edge: float = 2.0,
    min_conf: str = "MODERATE",
    start_bankroll: float = 1000,
    stake_method: str = "Flat (1%)",
    bet_odds: float = -110,
    scenario_probability: float = 0.50,
) -> dict:
    return value_bets.build_value_bets(
        season=season, bet_type=bet_type, min_edge=min_edge, min_conf=min_conf,
        start_bankroll=start_bankroll, stake_method=stake_method,
        bet_odds=bet_odds, scenario_probability=scenario_probability,
    )


@app.get("/api/team-explorer")
def api_team_explorer(team: str | None = None, season: int | None = None) -> dict:
    return team_explorer.build_team_explorer(team=team, season=season)


@app.get("/api/historical-analysis")
def api_historical(
    season_from: int | None = None,
    season_to: int | None = None,
    team_a: str | None = None,
    team_b: str | None = None,
) -> dict:
    return historical.build_historical(
        season_from=season_from, season_to=season_to, team_a=team_a, team_b=team_b
    )


@app.get("/api/model-performance")
def api_model_performance() -> dict:
    return model_performance.build_model_performance()


@app.get("/api/win-probability")
def api_win_probability(
    season: int = 2025,
    season_type: str = "Regular",
    week: int = 1,
    search: str = "",
    game_id: int | None = None,
) -> dict:
    return win_probability.build_win_probability(
        season=season, season_type_display=season_type, week=week,
        search=search, game_id=game_id,
    )


@app.get("/api/preseason-outlook")
def api_preseason(season: int | None = None, conference: str = "All") -> dict:
    return preseason.build_preseason(season=season, conference=conference)


@app.get("/api/data-quality")
def api_data_quality() -> dict:
    return data_quality.build_data_quality()


@app.get("/api/total-market-signals")
def api_total_signals(season: int | None = None) -> dict:
    return total_signals.build_total_signals(season=season)


# ---------------------------------------------------------------------------
# Static React build (production).  In development Vite serves the client and
# proxies /api here, so this mount simply does not exist.
# ---------------------------------------------------------------------------
if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        index = FRONTEND_DIST / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="Frontend build not found")
        return FileResponse(index)
else:

    @app.get("/", include_in_schema=False)
    def root_placeholder() -> JSONResponse:
        return JSONResponse(
            {
                "detail": "Frontend build not found. Run `npm run build` in frontend/.",
                "api_docs": "/docs",
            }
        )
