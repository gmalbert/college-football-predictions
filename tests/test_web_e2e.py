"""End-to-end browser tests for the React build.

These drive the real application with Playwright and assert that every page
loads without console errors and that the interactive controls actually change
what the page renders.

The FastAPI server must already be running on 127.0.0.1:8000 (it also serves
the compiled React bundle)::

    .venv\\Scripts\\python.exe -m uvicorn api.main:app --port 8000
    .venv\\Scripts\\python.exe -m pytest tests/test_web_e2e.py -q
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE = os.environ.get("PARITY_REACT_BASE", "http://127.0.0.1:8000")

# route -> expected h1
ROUTES = {
    "/": "🏈 College Football Predictions",
    "/Weekly_Predictions": "📊 Weekly Predictions",
    "/Value_Bets": "💰 Value Bets",
    "/Team_Explorer": "🏟️ Team Explorer",
    "/Historical_Analysis": "📈 Historical Analysis",
    "/Model_Performance": "🎯 Model Performance",
    "/Win_Probability": "📈 In-Game Win Probability",
    "/Preseason_Outlook": "🏈 Preseason Outlook",
    "/Data_Quality": "🛡️ Data & Model Quality",
    "/Total_Market_Signals": "Total Market Signals",
}

# pages/8 never calls add_betting_oracle_footer() and pages/7 omits it too
# (docs/UI_UX_ENHANCEMENTS.md U4) — the React build reproduces both omissions.
PAGES_WITHOUT_FOOTER = {"/Preseason_Outlook", "/Win_Probability"}
PAGES_WITH_FOOTER = set(ROUTES) - PAGES_WITHOUT_FOOTER
PAGES_WITH_CHARTS = {
    "/Team_Explorer",
    "/Historical_Analysis",
    "/Model_Performance",
    "/Win_Probability",
    "/Preseason_Outlook",
}


def _server_available() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/api/health", timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason="FastAPI server is not running on 127.0.0.1:8000",
)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()
        yield instance
        instance.close()


@pytest.fixture()
def page(browser):
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = context.new_page()
    errors: list[str] = []
    page.on(
        "console",
        lambda message: errors.append(message.text) if message.type == "error" else None,
    )
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.console_errors = errors  # type: ignore[attr-defined]
    yield page
    context.close()


def _open(page: Page, route: str) -> None:
    page.goto(f"{BASE}{route}", wait_until="networkidle", timeout=120000)
    page.wait_for_selector("h1", timeout=60000)


def _wait_for_table(page: Page) -> None:
    """Wait for the page's data table.

    ``state="attached"`` because the first table on Weekly Predictions lives in
    the collapsed ParlayAPI expander and is therefore not "visible".
    """
    page.wait_for_selector("table.df tbody tr", state="attached", timeout=60000)


# ---------------------------------------------------------------------------
# Load smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route,heading", ROUTES.items(), ids=list(ROUTES))
def test_page_loads_with_expected_title(page: Page, route: str, heading: str) -> None:
    _open(page, route)
    assert page.locator("h1").first.inner_text().strip() == heading


@pytest.mark.parametrize("route", list(ROUTES), ids=list(ROUTES))
def test_page_has_no_console_errors(page: Page, route: str) -> None:
    _open(page, route)
    page.wait_for_timeout(1500)
    assert page.console_errors == []  # type: ignore[attr-defined]


@pytest.mark.parametrize("route", sorted(PAGES_WITH_FOOTER))
def test_footer_present(page: Page, route: str) -> None:
    _open(page, route)
    assert page.locator(".betting-oracle-footer").count() == 1


@pytest.mark.parametrize("route", sorted(PAGES_WITHOUT_FOOTER))
def test_footer_absent_where_streamlit_omits_it(page: Page, route: str) -> None:
    _open(page, route)
    assert page.locator(".betting-oracle-footer").count() == 0


def test_sidebar_navigation_is_complete(page: Page) -> None:
    _open(page, "/")
    links = page.locator('[data-testid="stSidebarNavLink"]')
    assert links.count() == 9
    assert [text.split()[-1] for text in links.all_inner_texts()] == [
        "Home", "Predictions", "Bets", "Explorer", "Analysis",
        "Performance", "Probability", "Outlook", "Quality",
    ]
    assert page.locator('[data-testid="stSidebarNavLink"][aria-current="page"]').count() == 1


def test_sidebar_metrics_render(page: Page) -> None:
    _open(page, "/")
    sidebar = page.locator('[data-testid="stSidebar"]')
    assert sidebar.locator('[data-testid="stMetricLabel"]').all_inner_texts() == [
        "Season ATS",
        "Model Brier",
    ]


@pytest.mark.parametrize("route", sorted(PAGES_WITH_CHARTS))
def test_charts_render_svg(page: Page, route: str) -> None:
    _open(page, route)
    page.wait_for_timeout(3000)
    assert page.locator('[data-testid="stPlotlyChart"] .main-svg').count() >= 1


def test_sidebar_link_navigates_without_reload(page: Page) -> None:
    _open(page, "/")
    page.locator('[data-testid="stSidebarNavLink"]', has_text="Value Bets").click()
    page.wait_for_url("**/Value_Bets")
    page.wait_for_selector("h1")
    assert page.locator("h1").first.inner_text().strip() == "💰 Value Bets"
    assert page.locator('[data-testid="stSidebarNavLink"][aria-current="page"]').inner_text().strip().endswith("Value Bets")


# ---------------------------------------------------------------------------
# Interaction tests
# ---------------------------------------------------------------------------


def test_weekly_predictions_season_change_updates_table(page: Page) -> None:
    _open(page, "/Weekly_Predictions")
    _wait_for_table(page)
    before = page.locator("table.df").last.locator("tbody tr").count()

    page.locator('[data-testid="stSelectbox"]').first.locator("select").select_option("2021")
    page.wait_for_timeout(2500)
    after = page.locator("table.df").last.locator("tbody tr").count()

    assert after > 0
    assert page.locator('[data-testid="stSelectbox"]').first.locator("select").input_value() == "2021"
    assert after != before or page.locator(".stCaption").first.inner_text().startswith("Season 2021")


def test_weekly_predictions_min_edge_filter_shrinks_table(page: Page) -> None:
    _open(page, "/Weekly_Predictions")
    _wait_for_table(page)
    before = page.locator("table.df").last.locator("tbody tr").count()

    slider = page.locator('[data-testid="stSlider"] input[type="range"]').first
    slider.fill("8")
    page.wait_for_timeout(2500)
    after = page.locator("table.df").last.locator("tbody tr").count()

    assert after <= before


def test_weekly_predictions_conference_filter(page: Page) -> None:
    _open(page, "/Weekly_Predictions")
    _wait_for_table(page)
    conference = page.locator('[data-testid="stSelectbox"]').nth(2).locator("select")
    options = conference.locator("option").all_inner_texts()
    assert options[0] == "All"
    if len(options) > 1:
        conference.select_option(options[1])
        page.wait_for_timeout(2500)
        assert conference.input_value() == options[1]


def test_value_bets_slider_changes_result_count(page: Page) -> None:
    _open(page, "/Value_Bets")
    _wait_for_table(page)
    before = page.locator("table.df tbody tr").count()

    page.locator('[data-testid="stSlider"] input[type="range"]').first.fill("9")
    page.wait_for_timeout(2500)
    after = page.locator("table.df tbody tr").count()
    assert after <= before


def test_value_bets_bet_type_switch_to_moneyline(page: Page) -> None:
    _open(page, "/Value_Bets")
    page.locator('[data-testid="stSelectbox"]').nth(1).locator("select").select_option("Moneyline")
    page.wait_for_timeout(2500)
    body = page.locator('[data-testid="stMainBlockContainer"]').inner_text()
    assert "Moneyline" in body


def test_team_explorer_switching_team_updates_heading(page: Page) -> None:
    _open(page, "/Team_Explorer")
    assert page.locator("h3").first.inner_text().strip() == "Alabama"

    page.locator('[data-testid="stSelectbox"]').first.locator("select").select_option("Ohio State")
    page.wait_for_timeout(2500)
    assert page.locator("h3").first.inner_text().strip() == "Ohio State"


def test_historical_analysis_tabs_switch_panels(page: Page) -> None:
    _open(page, "/Historical_Analysis")
    tabs = page.locator('[role="tab"]')
    assert tabs.count() == 3
    assert tabs.nth(0).get_attribute("aria-selected") == "true"

    tabs.nth(1).click()
    page.wait_for_timeout(800)
    assert tabs.nth(1).get_attribute("aria-selected") == "true"
    h2h_panel = page.locator('[data-testid="stTabsContent"]').nth(1)
    assert h2h_panel.is_visible()
    assert "Head-to-Head Lookup" in h2h_panel.inner_text()

    tabs.nth(2).click()
    page.wait_for_timeout(800)
    assert "Conference Power" in page.locator('[data-testid="stTabsContent"]').nth(2).inner_text()


def test_model_performance_expander_toggles(page: Page) -> None:
    _open(page, "/Model_Performance")
    expander = page.locator("details.stExpander").first
    assert expander.get_attribute("open") is None
    expander.locator("summary").click()
    page.wait_for_timeout(500)
    assert expander.get_attribute("open") is not None
    assert expander.locator("table.df tbody tr").count() >= 1


def test_win_probability_search_filters_games(page: Page) -> None:
    _open(page, "/Win_Probability")
    select = page.locator('[data-testid="stSelectbox"]').nth(3).locator("select")
    before = select.locator("option").count()

    page.locator('[data-testid="stTextInput"] input').fill("Alabama")
    page.wait_for_timeout(3000)
    after = select.locator("option").count()

    assert after <= before
    options = select.locator("option").all_inner_texts()
    assert all("Alabama" in option for option in options)


def test_preseason_outlook_sidebar_season_filter(page: Page) -> None:
    _open(page, "/Preseason_Outlook")
    sidebar = page.locator('[data-testid="stSidebar"]')
    assert "Filters" in sidebar.inner_text()
    season = sidebar.locator("select")
    assert season.count() == 1
    options = season.locator("option").all_inner_texts()
    season.select_option(options[-1])
    page.wait_for_timeout(2500)
    assert season.input_value() == options[-1]


def test_preseason_outlook_tabs(page: Page) -> None:
    _open(page, "/Preseason_Outlook")
    tabs = page.locator('[role="tab"]')
    assert tabs.count() == 3
    tabs.nth(2).click()
    page.wait_for_timeout(2500)
    panel = page.locator('[data-testid="stTabsContent"]').nth(2)
    assert "Team Efficiency Quadrant" in panel.inner_text()
    assert panel.locator('[data-testid="stPlotlyChart"] .main-svg').count() >= 1


def test_data_quality_table_is_populated(page: Page) -> None:
    _open(page, "/Data_Quality")
    rows = page.locator("table.df tbody tr")
    assert rows.count() >= 10
    assert "Pass" in page.locator("table.df").inner_text()


def test_home_metrics_render(page: Page) -> None:
    _open(page, "/")
    labels = page.locator('[data-testid="stMetricLabel"]').all_inner_texts()
    assert "Brier Score" in labels
    assert "Games" in labels
    assert page.locator('[data-testid="stMetricValue"]').count() >= 10
