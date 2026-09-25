"""Generate the canonical per-page content signature from the Streamlit app.

``streamlit.testing.v1.AppTest`` executes each page in-process and exposes the
resulting element tree, including the *full* contents of every ``st.dataframe``
(which the browser renders onto a canvas and therefore cannot be scraped).

The output of this script is the ground truth that
``scripts/parity_check.py`` compares the React build against.

Usage::

    python scripts/parity_snapshot.py            # writes parity/expected/*.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_DIR = PROJECT_ROOT / "parity" / "expected"

# page file -> url slug used by both st.navigation and the React router
PAGES: dict[str, str] = {
    "predictions.py": "home",
    "pages/1_Weekly_Predictions.py": "Weekly_Predictions",
    "pages/2_Value_Bets.py": "Value_Bets",
    "pages/3_Team_Explorer.py": "Team_Explorer",
    "pages/4_Historical_Analysis.py": "Historical_Analysis",
    "pages/5_Model_Performance.py": "Model_Performance",
    "pages/7_Win_Probability.py": "Win_Probability",
    "pages/8_Preseason_Outlook.py": "Preseason_Outlook",
    "pages/9_Data_Quality.py": "Data_Quality",
    "pages/10_Total_Market_Signals.py": "Total_Market_Signals",
}

_MARKDOWN_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_HEADING_LINE = re.compile(r"^(#{1,3})\s+(.*)$")
_RULE_LINE = re.compile(r"^\s*(?:---|___|\*\*\*)\s*$")


def _clean(value) -> str:
    """Collapse whitespace the way ``innerText`` does in the browser."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _strip_markdown_tables(text: str) -> str:
    """Drop pipe-table rows, keeping line structure intact for heading parsing."""
    lines = [line for line in str(text).splitlines() if not _MARKDOWN_TABLE_ROW.match(line)]
    return "\n".join(lines)


def _split_markdown(text: str) -> tuple[dict[str, list[str]], str]:
    """Split a ``st.markdown`` block into heading levels and remaining prose.

    ``predictions.py`` renders its page title with ``st.markdown("# …")`` rather
    than ``st.title()``, so the AppTest markdown element has to be mined for the
    heading Streamlit actually renders.
    """
    headings: dict[str, list[str]] = {"titles": [], "headers": [], "subheaders": []}
    prose: list[str] = []
    for line in str(text).splitlines():
        if _RULE_LINE.match(line) or not line.strip():
            continue
        match = _HEADING_LINE.match(line.strip())
        if match:
            level = len(match.group(1))
            key = {1: "titles", 2: "headers", 3: "subheaders"}[level]
            headings[key].append(_clean(match.group(2)))
            continue
        prose.append(line)
    return headings, _clean(" ".join(prose))


def _dataframe_signature(frame) -> dict:
    columns = [str(column) for column in frame.columns]
    rows = [
        [None if value is None else value for value in row]
        for row in frame.astype(object).where(frame.notna(), None).values.tolist()
    ]
    return {"columns": columns, "rows": rows}


def snapshot_page(page_file: str) -> dict:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / page_file), default_timeout=300)
    app.run()

    exceptions = [str(exception.value) for exception in (app.exception or [])]

    markdown: list[str] = []
    extra_headings: dict[str, list[str]] = {"titles": [], "headers": [], "subheaders": []}
    for element in app.markdown:
        text = str(element.value or "")
        if text.lstrip().startswith("<style"):
            continue  # the theme CSS injected by apply_theme()
        headings, prose = _split_markdown(_strip_markdown_tables(text))
        for key, values in headings.items():
            extra_headings[key].extend(values)
        if prose:
            markdown.append(prose)

    def _ordered(elements) -> list[str]:
        """Heading elements Streamlit emitted directly."""
        return [_clean(element.value) for element in elements]

    return {
        "page_file": page_file,
        "exceptions": exceptions,
        "titles": _ordered(app.title) + extra_headings["titles"],
        "subheaders": _ordered(app.subheader) + extra_headings["subheaders"],
        "headers": _ordered(app.header) + extra_headings["headers"],
        "metrics": [
            {
                "label": _clean(element.label),
                "value": _clean(element.value),
                "delta": _clean(element.delta) if element.delta is not None else None,
            }
            for element in app.metric
        ],
        "captions": [_clean(element.value) for element in app.caption],
        "alerts": (
            [{"kind": "error", "text": _clean(element.value)} for element in app.error]
            + [{"kind": "warning", "text": _clean(element.value)} for element in app.warning]
            + [{"kind": "info", "text": _clean(element.value)} for element in app.info]
            + [{"kind": "success", "text": _clean(element.value)} for element in app.success]
        ),
        "markdown": markdown,
        "expanders": [_clean(getattr(element, "label", "")) for element in app.expander],
        "dataframes": [
            _dataframe_signature(element.value) for element in app.dataframe
        ],
    }


def main() -> int:
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for page_file, slug in PAGES.items():
        try:
            payload = snapshot_page(page_file)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAIL  {slug:22s} {type(exc).__name__}: {exc}")
            failures += 1
            continue
        destination = EXPECTED_DIR / f"{slug}.json"
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        status = "EXC" if payload["exceptions"] else "OK "
        print(
            f"{status}   {slug:22s} metrics={len(payload['metrics']):>3} "
            f"tables={len(payload['dataframes']):>2} "
            f"captions={len(payload['captions']):>3} "
            f"alerts={len(payload['alerts']):>2} -> {destination.name}"
        )
        if payload["exceptions"]:
            for message in payload["exceptions"]:
                print(f"        exception: {message[:160]}")
    print(f"\nwrote {len(PAGES) - failures} snapshots to {EXPECTED_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
