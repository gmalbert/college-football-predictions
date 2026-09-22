"""API contract tests for the FastAPI parity backend.

Every endpoint is exercised in-process through ``TestClient`` so the suite does
not need a running server.  A byte-compilation sweep over the whole repository
is included as well: the Streamlit app must keep importing cleanly after the
React/FastAPI additions.

Run with::

    .venv\\Scripts\\python.exe -m pytest tests/test_api_parity.py -q
"""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.data import cache_clear, cache_stats  # noqa: E402
from api.main import NAV_SECTIONS, app  # noqa: E402
from api.settings import load_settings, settings  # noqa: E402

client = TestClient(app)

# The nine pages registered in predictions.py's st.navigation, plus the
# unregistered page 10.
PAGE_ENDPOINTS = [
    ("/api/home", {}),
    ("/api/weekly-predictions", {"tz": "UTC"}),
    ("/api/value-bets", {}),
    ("/api/team-explorer", {}),
    ("/api/historical-analysis", {}),
    ("/api/model-performance", {}),
    ("/api/preseason-outlook", {}),
    ("/api/data-quality", {}),
    ("/api/total-market-signals", {}),
]


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_meta_navigation_matches_streamlit() -> None:
    """The sidebar must reproduce predictions.py::nav_sections exactly."""
    payload = client.get("/api/meta").json()
    sections = {entry["section"]: entry["pages"] for entry in payload["nav"]}

    assert list(sections) == ["", "Analysis"]
    assert [page["title"] for page in sections[""]] == ["Home"]
    assert [page["title"] for page in sections["Analysis"]] == [
        "Weekly Predictions",
        "Value Bets",
        "Team Explorer",
        "Historical Analysis",
        "Model Performance",
        "Win Probability",
        "Preseason Outlook",
        "Data & Model Quality",
    ]
    assert [page["path"] for page in sections["Analysis"]] == [
        "/Weekly_Predictions",
        "/Value_Bets",
        "/Team_Explorer",
        "/Historical_Analysis",
        "/Model_Performance",
        "/Win_Probability",
        "/Preseason_Outlook",
        "/Data_Quality",
    ]
    # Page 10 is not registered in st.navigation.
    assert all("Total_Market_Signals" not in page["path"] for page in sections["Analysis"])


def test_meta_exposes_sidebar_metrics_and_branding() -> None:
    payload = client.get("/api/meta").json()
    labels = [metric["label"] for metric in payload["sidebar"]["metrics"]]
    assert labels == ["Season ATS", "Model Brier"]
    assert payload["theme"]["primaryColor"] == "#2B7CB8"
    assert payload["footer"]["url"] == "https://www.betting-oracle.com"


def test_logo_is_served() -> None:
    response = client.get("/api/logo")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("path,params", PAGE_ENDPOINTS)
def test_page_endpoint_returns_json(path: str, params: dict) -> None:
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    assert "page" in payload
    assert "title" in payload
    # No NaN/Infinity leaked into the JSON.
    assert "NaN" not in response.text
    assert "Infinity" not in response.text


def test_home_columns_match_streamlit_headings() -> None:
    payload = client.get("/api/home").json()
    assert payload["title"] == "🏈 College Football Predictions"
    assert [column["heading"] for column in payload["columns"]] == [
        "Upcoming Model Deltas",
        "📐 Model Accuracy",
        "📊 Dataset",
    ]


def test_weekly_predictions_shape() -> None:
    payload = client.get("/api/weekly-predictions", params={"tz": "UTC"}).json()
    assert payload["title"] == "📊 Weekly Predictions"
    assert payload["selection"]["conferences"][0] == "All"
    if not payload.get("stopped"):
        columns = payload["table"]["columns"]
        assert columns[0] == "Game"
        assert "Kickoff (UTC)" in columns
        assert payload["table_height"] >= 140


def test_weekly_timezone_changes_kickoff_column() -> None:
    utc = client.get("/api/weekly-predictions", params={"tz": "UTC"}).json()
    eastern = client.get(
        "/api/weekly-predictions", params={"tz": "America/New_York"}
    ).json()
    assert "Kickoff (UTC)" in utc["table"]["columns"]
    assert "Kickoff (ET)" in eastern["table"]["columns"]


def test_team_explorer_defaults_to_alabama() -> None:
    payload = client.get("/api/team-explorer").json()
    assert payload["selection"]["team"] == "Alabama"
    labels = [metric["label"] for metric in payload["metrics"]]
    assert labels == ["Record", "Conference", "SP+", "Talent", "Games"]


def test_historical_defaults_to_ohio_state_vs_michigan() -> None:
    payload = client.get("/api/historical-analysis").json()
    assert payload["selection"]["team_a"] == "Ohio State"
    assert payload["selection"]["team_b"] == "Michigan"
    assert [tab["label"] for tab in payload["tabs"]] == [
        "📅 Season Trends",
        "🆚 H2H Lookup",
        "🏆 Conference Power",
    ]
    assert payload["tabs"][1]["heading"] == "Head-to-Head Lookup"
    assert payload["tabs"][2]["heading"] == "Conference Power"


def test_model_performance_metrics() -> None:
    payload = client.get("/api/model-performance").json()
    labels = [metric["label"] for metric in payload["metrics"]]
    assert labels == [
        "OOS Brier Score",
        "OOS Spread RMSE",
        "OOS Total RMSE",
        "OOS ATS Win %",
        "ATS Record",
    ]
    assert payload["comparison"]["columns"][0] == "Target"
    assert len(payload["comparison"]["rows"]) == 3


def test_preseason_outlook_omits_footer_like_streamlit() -> None:
    """pages/8 never calls add_betting_oracle_footer() — parity is intentional."""
    payload = client.get("/api/preseason-outlook").json()
    assert payload["footer"] is False
    assert payload["sidebar"]["heading"] == "Filters"
    assert [tab["label"] for tab in payload["tabs"]] == [
        "📊 Returning Production",
        "🔄 Transfer Portal",
        "⚡ Team Efficiency Quadrant",
    ]


def test_data_quality_summary_counts() -> None:
    payload = client.get("/api/data-quality").json()
    assert [metric["label"] for metric in payload["metrics"]] == [
        "Passing",
        "Warnings",
        "Failures",
    ]
    assert payload["table"]["columns"] == ["Status", "Check", "Details", "Value"]


def test_total_market_signals_is_hold() -> None:
    payload = client.get("/api/total-market-signals").json()
    assert payload["errors"] == ["The total-side strategy is on hold."]
    assert payload["metrics"][0]["label"] == "OOS Brier"


def test_win_probability_controls() -> None:
    payload = client.get("/api/win-probability").json()
    assert payload["controls"]["seasons"] == [2025, 2024, 2023, 2022, 2021]
    assert payload["controls"]["season_types"] == ["Regular", "Postseason"]
    assert payload["controls"]["weeks"] == list(range(1, 18))


def test_response_time_header_is_recorded() -> None:
    response = client.get("/api/home")
    assert "X-Response-Time-Ms" in response.headers
    stats = client.get("/api/stats").json()
    assert "/api/home" in stats["endpoints"]


def test_health_reports_effective_configuration() -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert "config" in payload
    assert payload["config"]["auth"] in {"none", "api-key"}


def test_health_is_reachable_without_a_key() -> None:
    """Healthchecks and load balancers must not need a credential."""
    assert client.get("/api/health").status_code == 200


def test_cache_clear_is_disabled_without_an_admin_token() -> None:
    """The endpoint forces a multi-second rebuild, so it must not be public."""
    assert not settings.admin_enabled
    assert client.post("/api/cache/clear").status_code == 404


# ---------------------------------------------------------------------------
# Settings parsing
# ---------------------------------------------------------------------------


def test_settings_defaults() -> None:
    parsed = load_settings({})
    assert parsed.host == "127.0.0.1"
    assert parsed.port == 8000
    assert parsed.workers == 1
    assert parsed.cors_origins == ()
    assert parsed.cors_enabled is False
    assert parsed.auth_enabled is False
    assert parsed.admin_enabled is False
    assert parsed.warm_cache is True


def test_settings_parse_origins() -> None:
    parsed = load_settings(
        {"TAILGATE_CORS_ORIGINS": "https://app.example.com, https://other.example.com/"}
    )
    assert parsed.cors_origins == (
        "https://app.example.com",
        "https://other.example.com",
    )
    assert parsed.cors_enabled is True


def test_settings_blank_origins_disable_cors() -> None:
    assert load_settings({"TAILGATE_CORS_ORIGINS": "  "}).cors_enabled is False


@pytest.mark.parametrize(
    "raw,expected",
    [("true", True), ("TRUE", True), ("1", True), ("on", True), ("yes", True),
     ("false", False), ("0", False), ("off", False), ("", False)],
)
def test_settings_parse_booleans(raw: str, expected: bool) -> None:
    assert load_settings({"TAILGATE_WARM_CACHE": raw}).warm_cache is expected


@pytest.mark.parametrize("raw", ["not-a-number", "-5", ""])
def test_settings_reject_bad_numbers(raw: str) -> None:
    parsed = load_settings({"TAILGATE_PORT": raw, "TAILGATE_WORKERS": raw})
    assert parsed.port == 8000
    assert parsed.workers == 1


def test_settings_blank_secrets_are_treated_as_unset() -> None:
    parsed = load_settings({"TAILGATE_API_KEY": "  ", "TAILGATE_ADMIN_TOKEN": ""})
    assert parsed.api_key is None
    assert parsed.admin_token is None
    assert parsed.auth_enabled is False
    assert parsed.admin_enabled is False


def test_settings_describe_redacts_secrets() -> None:
    described = load_settings(
        {"TAILGATE_API_KEY": "super-secret", "TAILGATE_ADMIN_TOKEN": "other-secret"}
    ).describe()
    assert "super-secret" not in json.dumps(described)
    assert "other-secret" not in json.dumps(described)
    assert described["auth"] == "api-key"
    assert described["admin_endpoints"] == "enabled"


def test_multi_worker_warns_about_per_process_cache() -> None:
    notes = load_settings({"TAILGATE_WORKERS": "4"}).warnings()
    assert any("each worker holds its own" in note for note in notes)


def _reload_app(monkeypatch: pytest.MonkeyPatch, **env: str):
    """Rebuild the app with ``env`` applied.

    ``api.main`` does ``from api.settings import settings``, so the settings
    module has to be reloaded first or the cached ``settings`` object is reused
    and the new environment is ignored.
    """
    import importlib

    import api.settings as settings_module
    import api.main as main_module

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    importlib.reload(settings_module)
    return importlib.reload(main_module)


def _restore(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    import importlib

    import api.settings as settings_module
    import api.main as main_module

    for name in names:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(settings_module)
    importlib.reload(main_module)


def test_api_key_is_enforced_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _reload_app(monkeypatch, TAILGATE_API_KEY="sekrit")
    try:
        secured = TestClient(reloaded.app)
        # Healthchecks stay open so a load balancer needs no credential.
        assert secured.get("/api/health").status_code == 200
        # Everything else under /api is rejected without the key.
        assert secured.get("/api/home").status_code == 401
        assert secured.get("/api/home", headers={"X-API-Key": "wrong"}).status_code == 401
        # Both header forms are accepted.
        assert secured.get("/api/home", headers={"X-API-Key": "sekrit"}).status_code == 200
        assert (
            secured.get("/api/home", headers={"Authorization": "Bearer sekrit"}).status_code
            == 200
        )
    finally:
        _restore(monkeypatch, "TAILGATE_API_KEY")


def test_admin_token_gates_cache_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _reload_app(monkeypatch, TAILGATE_ADMIN_TOKEN="admin-sekrit")
    try:
        admin = TestClient(reloaded.app)
        assert admin.post("/api/cache/clear").status_code == 403
        assert (
            admin.post("/api/cache/clear", headers={"X-Admin-Token": "nope"}).status_code == 403
        )
        assert (
            admin.post("/api/cache/clear", headers={"X-Admin-Token": "admin-sekrit"}).json()
            == {"status": "cleared"}
        )
    finally:
        _restore(monkeypatch, "TAILGATE_ADMIN_TOKEN")


def test_cors_middleware_only_when_origins_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    allowed = "https://app.example.com"

    reloaded = _reload_app(monkeypatch, TAILGATE_CORS_ORIGINS=allowed)
    try:
        cors_client = TestClient(reloaded.app)
        preflight = cors_client.options(
            "/api/home",
            headers={"Origin": allowed, "Access-Control-Request-Method": "GET"},
        )
        assert preflight.headers.get("access-control-allow-origin") == allowed
    finally:
        _restore(monkeypatch, "TAILGATE_CORS_ORIGINS")

    # With no origins configured the middleware is absent, so no CORS headers.
    response = client.get("/api/home", headers={"Origin": allowed})
    assert "access-control-allow-origin" not in response.headers


def test_cache_clear_roundtrip() -> None:
    """The cache can still be cleared in-process (used by the benchmark)."""
    client.get("/api/home")
    cache_clear()
    assert cache_stats()["entries"] == 0


# ---------------------------------------------------------------------------
# Byte-compilation sweep
# ---------------------------------------------------------------------------


def _python_files() -> list[Path]:
    roots = ["api", "utils", "models", "scripts", "tests"]
    files: list[Path] = [PROJECT_ROOT / "predictions.py", PROJECT_ROOT / "footer.py"]
    for root in roots:
        files.extend(sorted((PROJECT_ROOT / root).rglob("*.py")))
    files.extend(sorted((PROJECT_ROOT / "pages").glob("*.py")))
    return [path for path in files if "__pycache__" not in path.parts]


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_module_byte_compiles(path: Path) -> None:
    py_compile.compile(str(path), doraise=True, cfile=None)
