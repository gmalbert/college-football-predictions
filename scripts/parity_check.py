"""Compare the React build against the Streamlit ground-truth snapshots.

Two independent verifications run here:

``--content``
    Drives the React app with Playwright and compares the rendered DOM against
    ``parity/expected/*.json`` (produced by ``scripts/parity_snapshot.py`` from
    ``streamlit.testing.v1.AppTest``).  This checks every heading, metric,
    caption, alert and the *complete* contents of every data table.

``--live``
    Loads the live Streamlit server and the live React server in the same
    browser and compares the structural chrome that AppTest cannot observe:
    sidebar navigation, widget labels, chart count/titles and expander labels.

Usage::

    python scripts/parity_snapshot.py     # refresh ground truth
    python scripts/parity_check.py --content --live --json
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DIR = PROJECT_ROOT / "parity" / "expected"
REPORT_PATH = PROJECT_ROOT / "parity" / "parity_report.json"

PAGES = [
    "home",
    "Weekly_Predictions",
    "Value_Bets",
    "Team_Explorer",
    "Historical_Analysis",
    "Model_Performance",
    "Win_Probability",
    "Preseason_Outlook",
    "Data_Quality",
    "Total_Market_Signals",
]

REACT_BASE = os.environ.get("PARITY_REACT_BASE", "http://127.0.0.1:8000")
STREAMLIT_BASE = os.environ.get("PARITY_STREAMLIT_BASE", "http://127.0.0.1:8501")

# The reference app must be Tailgate Edge and nothing else. Port 8501 is a
# common default and another project's Streamlit app can occupy it, which would
# otherwise make every live comparison fail for reasons that have nothing to do
# with this repo. These are checked before any comparison runs.
REACT_MARKER = "Weekly Predictions"
STREAMLIT_MARKER = "Weekly Predictions"

# ``pages/10_Total_Market_Signals.py`` is not registered in
# ``predictions.py::nav_sections``, so the live Streamlit server answers 404 for
# it (docs/UI_UX_ENHANCEMENTS.md U1).  The React build keeps the route so the
# page is not lost, but it is absent from the sidebar in both apps.
STREAMLIT_404_PAGES = {"Total_Market_Signals"}

# ---------------------------------------------------------------------------
# Shared extraction (executed inside the browser)
# ---------------------------------------------------------------------------

_EXTRACT_BODY = r"""
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  // innerText respects visibility and inserts real whitespace; textContent is
  // the fallback for nodes inside a collapsed <details> or a hidden tab panel.
  const text = (e) => { if (!e) return ''; const visible = clean(e.innerText); return visible || clean(e.textContent); };
  const q = (sel) => Array.from(document.querySelectorAll(sel));
  const main = (e) => !!e.closest('[data-testid="stMainBlockContainer"], [data-testid="stMain"], .main');
  const skip = ['[data-testid="stMetric"]','[data-testid="stCaptionContainer"]','[data-testid="stAlert"]',
                '[data-testid="stWidgetLabel"]','[data-testid="stSelectbox"]','[data-testid="stSlider"]',
                '[data-testid="stDataFrame"]','table','[data-testid="stExpander"] > summary'];
  const markdown = q('[data-testid="stMarkdownContainer"]')
    .filter(main)
    .filter((e) => !skip.some((sel) => e.closest(sel)))
    .map(text).filter(Boolean);
  const cells = (root, sel) => Array.from(root.querySelectorAll(sel)).map((c) => clean(c.textContent));
  return {
    titles: q('h1').map(text),
    subheaders: q('h3').map(text),
    headers: q('h2').map(text),
    metrics: q('[data-testid="stMetric"]').map((e) => ({
      label: text(e.querySelector('[data-testid="stMetricLabel"]')),
      value: text(e.querySelector('[data-testid="stMetricValue"]')),
      delta: e.querySelector('[data-testid="stMetricDelta"]') ? text(e.querySelector('[data-testid="stMetricDelta"]')) : null,
    })),
    captions: q('[data-testid="stCaptionContainer"]').map(text),
    alerts: q('[data-testid="stAlert"]').map((e) => ({ kind: ALERT_KIND(e), text: text(e) })),
    markdown: markdown,
    expanders: q(EXPANDER_SELECTOR).map(text),
    dataframes: q('table.df, [data-testid="stDataFrame"]').filter((el) => el.tagName === 'TABLE').map((t) => ({
      columns: cells(t, 'thead th'),
      rows: Array.from(t.querySelectorAll('tbody tr')).map((tr) => cells(tr, 'td')),
    })),
    nav: q('[data-testid="stSidebarNavLink"]').map((e) => ({ text: text(e), current: e.getAttribute('aria-current') })),
    widget_labels: q('[data-testid="stWidgetLabel"]').map(text),
    chart_titles: q('[data-testid="stPlotlyChart"] .gtitle').map((e) => clean(e.textContent)).filter(Boolean),
    chart_count: q('[data-testid="stPlotlyChart"]').length,
    table_count: q('[data-testid="stDataFrame"]').length,
    has_footer: HAS_FOOTER,
  };
"""

EXTRACT_REACT = (
    "() => {\n"
    + _EXTRACT_BODY.replace(
        "ALERT_KIND(e)",
        "(e.dataset.alertKind || 'info')",
    )
    .replace("EXPANDER_SELECTOR", "'details.stExpander > summary'")
    .replace("HAS_FOOTER", "!!document.querySelector('.betting-oracle-footer')")
    + "\n}"
)

EXTRACT_STREAMLIT = (
    "() => {\n"
    + """
  const kindOf = (e) => {
    for (const k of ['Info','Warning','Error','Success']) {
      if (e.querySelector(`[data-testid="stAlertContent${k}"]`)) return k.toLowerCase();
    }
    return 'info';
  };
"""
    + _EXTRACT_BODY.replace("ALERT_KIND(e)", "kindOf(e)")
    .replace("EXPANDER_SELECTOR", "'[data-testid=\"stExpander\"] summary'")
    .replace(
        "HAS_FOOTER",
        "document.body.innerText.includes('Powered by Betting Oracle')",
    )
    + "\n}"
)

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_HOURS = re.compile(r"(\d+(?:\.\d+)?)\s*hours")
_ICON_PREFIX = re.compile(
    "^[\\s\u2000-\u3300\ufe0f\u2600-\u27bf\u2b00-\u2bff\u2190-\u21ff\U0001f000-\U0001faff]+"
)


def strip_icon(text: str) -> str:
    """Drop a leading status emoji (React renders one, Streamlit uses a font icon)."""
    return _ICON_PREFIX.sub("", text or "").strip()


def strip_markdown(text: str) -> str:
    """Reduce Markdown source to the plain text a browser would display."""
    value = str(text or "")
    if "<" in value and ">" in value:
        value = re.sub(r"<[^>]+>", " ", value)
        value = html.unescape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"\1", value, flags=re.S)
    value = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", value)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"^\s*#{1,6}\s*", "", value)
    value = re.sub(r"^\s*---\s*$", "", value, flags=re.M)
    return re.sub(r"\s+", " ", value).strip()


def normalise_volatile(text: str) -> str:
    """Round wall-clock-derived ages so two runs minutes apart still compare."""
    return _HOURS.sub(lambda m: f"{round(float(m.group(1)) / 10) * 10:.0f} hours", text)


def normalise_cell(value) -> str:
    """Make a table cell comparable across the pandas and DOM renderers.

    Streamlit's ``ProgressColumn`` renders ``62.3%`` where the underlying value
    is ``62.3``; both sides are reduced to the bare number so the comparison
    stays strict about the digits while tolerating the percent decoration.
    """
    if value is None:
        return ""
    text = normalise_volatile(re.sub(r"\s+", " ", str(value)).strip())
    if text.endswith("%") and _NUMBER.match(text[:-1]):
        return text[:-1]
    if _NUMBER.match(text):
        try:
            number = float(text)
        except ValueError:
            return text
        return str(int(number)) if number.is_integer() else f"{number:.6f}".rstrip("0").rstrip(".")
    return text


def cells_equal(left: str, right: str) -> bool:
    """Numeric cells may differ slightly when they are derived from ``now()``."""
    if left == right:
        return True
    if _NUMBER.match(left) and _NUMBER.match(right):
        a, b = float(left), float(right)
        return abs(a - b) <= max(0.05, 0.002 * max(abs(a), abs(b)))
    return False


def rows_equal(left: list[str], right: list[str]) -> bool:
    return len(left) == len(right) and all(
        cells_equal(a, b) for a, b in zip(left, right)
    )


def normalise_table(table: dict) -> dict:
    return {
        "columns": [normalise_cell(c) for c in table.get("columns", [])],
        "rows": [[normalise_cell(c) for c in row] for row in table.get("rows", [])],
    }


def normalise_metric(metric: dict) -> tuple:
    delta = metric.get("delta")
    return (
        re.sub(r"\s+", " ", str(metric.get("label") or "")).strip(),
        re.sub(r"\s+", " ", str(metric.get("value") or "")).strip(),
        None if not delta else re.sub(r"\s+", " ", str(delta)).strip(),
    )


def normalise_alerts(alerts: list[dict]) -> list[tuple]:
    return sorted(
        (a.get("kind"), normalise_volatile(strip_icon(re.sub(r"\s+", " ", str(a.get("text") or "")).strip())))
        for a in alerts
    )


def strip_material_icon(text: str) -> str:
    """Streamlit renders expander chevrons as a Material icon ligature in the text."""
    return re.sub(
        r"^(?:keyboard_arrow_(?:right|down)|arrow_(?:right|down))\s*", "", text or ""
    ).strip()


def normalise_text(value) -> str:
    """Plain-text projection of a heading/caption for cross-renderer comparison."""
    return normalise_volatile(strip_markdown(re.sub(r"\s+", " ", str(value or "")).strip()))


def normalise_markdown(values: list[str]) -> list[str]:
    out = []
    for value in values:
        cleaned = strip_markdown(value)
        if cleaned:
            out.append(normalise_volatile(cleaned))
    return sorted(out)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def compare_content(expected: dict, actual: dict) -> list[str]:
    problems: list[str] = []

    for field in ("titles", "subheaders", "headers", "captions"):
        want = [normalise_text(v) for v in expected.get(field, [])]
        got = [normalise_text(v) for v in actual.get(field, [])]
        if want != got:
            problems.append(f"{field}:\n    expected {want}\n    actual   {got}")

    want_metrics = sorted(normalise_metric(m) for m in expected.get("metrics", []))
    got_metrics = sorted(normalise_metric(m) for m in actual.get("metrics", []))
    if want_metrics != got_metrics:
        problems.append(
            f"metrics:\n    only in streamlit {[m for m in want_metrics if m not in got_metrics]}"
            f"\n    only in react     {[m for m in got_metrics if m not in want_metrics]}"
        )

    want_alerts, got_alerts = normalise_alerts(expected.get("alerts", [])), normalise_alerts(actual.get("alerts", []))
    if want_alerts != got_alerts:
        problems.append(f"alerts:\n    expected {want_alerts}\n    actual   {got_alerts}")

    want_md, got_md = normalise_markdown(expected.get("markdown", [])), normalise_markdown(actual.get("markdown", []))
    if want_md != got_md:
        problems.append(
            f"markdown:\n    only in streamlit {[m for m in want_md if m not in got_md]}"
            f"\n    only in react     {[m for m in got_md if m not in want_md]}"
        )

    want_exp = sorted(strip_material_icon(e) for e in expected.get("expanders", []))
    got_exp = sorted(strip_material_icon(e) for e in actual.get("expanders", []))
    if want_exp != got_exp:
        problems.append(f"expanders:\n    expected {want_exp}\n    actual   {got_exp}")

    want_tables = [normalise_table(t) for t in expected.get("dataframes", [])]
    got_tables = [normalise_table(t) for t in actual.get("dataframes", [])]
    if len(want_tables) != len(got_tables):
        problems.append(f"table count: expected {len(want_tables)}, actual {len(got_tables)}")
    for index, (want, got) in enumerate(zip(want_tables, got_tables)):
        if want["columns"] != got["columns"]:
            problems.append(
                f"table[{index}] columns:\n    expected {want['columns']}\n    actual   {got['columns']}"
            )
            continue
        if len(want["rows"]) != len(got["rows"]):
            problems.append(
                f"table[{index}] row count: expected {len(want['rows'])}, actual {len(got['rows'])}"
            )
        for row_index, (want_row, got_row) in enumerate(zip(want["rows"], got["rows"])):
            if not rows_equal(want_row, got_row):
                problems.append(
                    f"table[{index}] row {row_index}:\n    expected {want_row}\n    actual   {got_row}"
                )
                break

    return problems


def compare_live(streamlit: dict, react: dict) -> list[str]:
    problems: list[str] = []

    for field in ("titles", "subheaders", "headers", "captions"):
        want, got = streamlit.get(field), react.get(field)
        if [normalise_text(v) for v in want or []] != [normalise_text(v) for v in got or []]:
            problems.append(f"{field}:\n    streamlit {want}\n    react     {got}")

    want_expanders = sorted(strip_material_icon(e) for e in streamlit.get("expanders", []))
    got_expanders = sorted(strip_material_icon(e) for e in react.get("expanders", []))
    if want_expanders != got_expanders:
        problems.append(
            f"expanders:\n    streamlit {want_expanders}\n    react     {got_expanders}"
        )

    for field in ("chart_titles", "chart_count", "table_count"):
        want, got = streamlit.get(field), react.get(field)
        if want != got:
            problems.append(f"{field}:\n    streamlit {want}\n    react     {got}")

    want_metrics = [normalise_metric(m) for m in streamlit.get("metrics", [])]
    got_metrics = [normalise_metric(m) for m in react.get("metrics", [])]
    if want_metrics != got_metrics:
        problems.append(f"metrics:\n    streamlit {want_metrics}\n    react     {got_metrics}")

    want_nav = [n["text"] for n in streamlit.get("nav", [])]
    got_nav = [n["text"] for n in react.get("nav", [])]
    if want_nav != got_nav:
        problems.append(f"nav:\n    streamlit {want_nav}\n    react     {got_nav}")

    want_current = [n["current"] for n in streamlit.get("nav", [])]
    got_current = [n["current"] for n in react.get("nav", [])]
    if want_current != got_current:
        problems.append(f"nav active:\n    streamlit {want_current}\n    react     {got_current}")

    want_labels = sorted(streamlit.get("widget_labels", []))
    got_labels = sorted(react.get("widget_labels", []))
    if want_labels != got_labels:
        problems.append(f"widget labels:\n    streamlit {want_labels}\n    react     {got_labels}")

    want_alerts, got_alerts = normalise_alerts(streamlit.get("alerts", [])), normalise_alerts(react.get("alerts", []))
    if want_alerts != got_alerts:
        problems.append(f"alerts:\n    streamlit {want_alerts}\n    react     {got_alerts}")

    if bool(streamlit.get("has_footer")) != bool(react.get("has_footer")):
        problems.append(
            f"footer: streamlit={streamlit.get('has_footer')} react={react.get('has_footer')}"
        )

    return problems


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def preflight(browser) -> list[str]:
    """Confirm both servers are actually running this application.

    Port 8501 is Streamlit's default and other projects use it too. Without this
    check a foreign app on the port produces a wall of parity failures that look
    like regressions but are not.
    """
    problems: list[str] = []
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    for label, base, marker in (
        ("React", REACT_BASE, REACT_MARKER),
        ("Streamlit", STREAMLIT_BASE, STREAMLIT_MARKER),
    ):
        try:
            page.goto(f"{base}/", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)
            body = page.evaluate("() => document.body.innerText")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{label} at {base} is unreachable: {type(exc).__name__}: {exc}")
            continue
        if marker not in body:
            head = " ".join(body.split())[:160]
            problems.append(
                f"{label} at {base} does not look like Tailgate Edge "
                f"(expected to find {marker!r}). Page starts with: {head!r}"
            )
        else:
            print(f"  preflight {label:<9} {base} ok")
    page.close()
    return problems


def run_content_check(browser, results: dict) -> None:
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    for slug in PAGES:
        expected_path = EXPECTED_DIR / f"{slug}.json"
        if not expected_path.exists():
            results[slug] = {"status": "missing-snapshot", "problems": []}
            continue
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        url = REACT_BASE + ("/" if slug == "home" else f"/{slug}")
        page.goto(f"{url}?tz=UTC", wait_until="networkidle", timeout=180000)
        page.wait_for_timeout(2500)
        actual = page.evaluate(EXTRACT_REACT)
        problems = compare_content(expected, actual)
        results[slug] = {
            "status": "pass" if not problems else "fail",
            "problems": problems,
            "counts": {
                "metrics": len(actual["metrics"]),
                "tables": len(actual["dataframes"]),
                "captions": len(actual["captions"]),
                "alerts": len(actual["alerts"]),
                "charts": actual["chart_count"],
            },
        }
    page.close()


def run_live_check(browser, results: dict) -> None:
    streamlit_page = browser.new_page(viewport={"width": 1600, "height": 1000})
    react_page = browser.new_page(viewport={"width": 1600, "height": 1000})
    for slug in PAGES:
        path = "/" if slug == "home" else f"/{slug}"
        streamlit_page.goto(f"{STREAMLIT_BASE}{path}", wait_until="networkidle", timeout=180000)
        streamlit_page.wait_for_timeout(6000)
        react_page.goto(f"{REACT_BASE}{path}", wait_until="networkidle", timeout=120000)
        react_page.wait_for_timeout(2500)

        entry = results.setdefault(slug, {})
        if slug in STREAMLIT_404_PAGES:
            not_found = streamlit_page.evaluate(
                "() => document.body.innerText.includes('Page not found')"
            )
            entry["live_status"] = "n/a" if not_found else "fail"
            entry["live_problems"] = [] if not_found else [
                "expected the live Streamlit server to 404 an unregistered page"
            ]
            entry["live_note"] = "orphaned page: Streamlit returns 404, React keeps the route unlinked"
            continue

        want = streamlit_page.evaluate(EXTRACT_STREAMLIT)
        got = react_page.evaluate(EXTRACT_REACT)
        problems = compare_live(want, got)
        entry["live_status"] = "pass" if not problems else "fail"
        entry["live_problems"] = problems
    streamlit_page.close()
    react_page.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", action="store_true", help="AppTest snapshot vs React DOM")
    parser.add_argument("--live", action="store_true", help="live Streamlit vs live React chrome")
    parser.add_argument("--json", action="store_true", help="write parity/parity_report.json")
    args = parser.parse_args()
    if not args.content and not args.live:
        args.content = args.live = True

    from playwright.sync_api import sync_playwright

    results: dict = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        print("=== preflight ===")
        blockers = preflight(browser)
        if blockers:
            browser.close()
            print("\nRefusing to compare — the servers are not serving this app:\n")
            for blocker in blockers:
                print(f"  - {blocker}")
            print(
                "\nStart the reference servers (see docs/FASTAPI_REACT_PARITY.md), or point the\n"
                "harness at different ports with PARITY_STREAMLIT_BASE / PARITY_REACT_BASE."
            )
            return 2
        if args.content:
            print("\n=== CONTENT PARITY (Streamlit AppTest  vs  React DOM) ===")
            run_content_check(browser, results)
        if args.live:
            print("\n=== LIVE CHROME PARITY (live Streamlit  vs  live React) ===")
            run_live_check(browser, results)
        browser.close()

    failures = 0
    for slug in PAGES:
        entry = results.get(slug, {})
        line = f"  {slug:22s}"
        if "status" in entry:
            line += f" content={entry['status']:5s}"
            failures += entry["status"] != "pass"
        if "live_status" in entry:
            line += f" live={entry['live_status']:5s}"
            failures += entry["live_status"] not in ("pass", "n/a")
        print(line)

    if args.json:
        REPORT_PATH.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nreport -> {REPORT_PATH}")
        print("(open the report to read exact strings; the console cannot render U+2011)")

    print(f"\nfailures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
