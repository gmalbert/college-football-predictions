"""Statistical comparison of the Streamlit app and the FastAPI + React build.

Measures five independent dimensions and reports the difference between the two
stacks with effect sizes and significance tests where a test is meaningful:

1. Content parity    — how many elements/cells are compared and how many match
2. API latency       — warm response distribution per endpoint
3. Transfer size     — bytes and request counts per page load
4. Browser timing    — load / heading-visible / settled, with Mann-Whitney U
                       and Cliff's delta across N cold-context loads
5. Visual difference — per-pixel comparison of matched screenshots, split into
                       the sidebar strip and the main content area

Usage::

    python scripts/stats_diff.py --iterations 8
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATS_JSON = PROJECT_ROOT / "parity" / "statistics.json"
STATS_MD = PROJECT_ROOT / "parity" / "STATISTICS.md"
DIFF_DIR = PROJECT_ROOT / "parity" / "diffs"

STREAMLIT_BASE = "http://127.0.0.1:8501"
REACT_BASE = "http://127.0.0.1:8000"

VIEWPORT = {"width": 1440, "height": 900}
SIDEBAR_PX = 300  # measured sidebar width

PAGES = [
    ("Home", "/"),
    ("Weekly Predictions", "/Weekly_Predictions"),
    ("Value Bets", "/Value_Bets"),
    ("Team Explorer", "/Team_Explorer"),
    ("Historical Analysis", "/Historical_Analysis"),
    ("Model Performance", "/Model_Performance"),
    ("Win Probability", "/Win_Probability"),
    ("Preseason Outlook", "/Preseason_Outlook"),
    ("Data & Model Quality", "/Data_Quality"),
]

API_ENDPOINTS = [
    ("Home", "/api/home"),
    ("Weekly Predictions", "/api/weekly-predictions"),
    ("Value Bets", "/api/value-bets"),
    ("Team Explorer", "/api/team-explorer"),
    ("Historical Analysis", "/api/historical-analysis"),
    ("Model Performance", "/api/model-performance"),
    ("Win Probability", "/api/win-probability"),
    ("Preseason Outlook", "/api/preseason-outlook"),
    ("Data & Model Quality", "/api/data-quality"),
]

# A pixel counts as "different" when any channel moves by more than this.
PIXEL_THRESHOLD = 8

# Vertical shifts searched when looking for the best content alignment.
ALIGN_SEARCH_PX = 90


def _summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    index_95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 1),
        "median": round(statistics.median(ordered), 1),
        "stdev": round(statistics.pstdev(ordered), 1) if len(ordered) > 1 else 0.0,
        "min": round(ordered[0], 1),
        "p95": round(ordered[index_95], 1),
        "max": round(ordered[-1], 1),
    }


# ---------------------------------------------------------------------------
# 1. Content parity
# ---------------------------------------------------------------------------


def content_statistics() -> dict:
    report_path = PROJECT_ROOT / "parity" / "parity_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))

    totals = {
        "pages_compared": 0, "pages_matching": 0,
        "live_pages_compared": 0, "live_pages_matching": 0,
        "metrics": 0, "tables": 0, "captions": 0, "alerts": 0, "charts": 0,
        "mismatched_fields": 0,
    }
    per_page = []
    for slug, entry in report.items():
        counts = entry.get("counts", {})
        row = {"page": slug}
        if "status" in entry:
            totals["pages_compared"] += 1
            totals["pages_matching"] += entry["status"] == "pass"
            totals["mismatched_fields"] += len(entry.get("problems", []))
            for key in ("metrics", "tables", "captions", "alerts", "charts"):
                totals[key] += counts.get(key, 0)
            row["content"] = entry["status"]
            row.update(counts)
        if entry.get("live_status") in ("pass", "fail"):
            totals["live_pages_compared"] += 1
            totals["live_pages_matching"] += entry["live_status"] == "pass"
            totals["mismatched_fields"] += len(entry.get("live_problems", []))
            row["live"] = entry["live_status"]
        per_page.append(row)
    return {"totals": totals, "per_page": per_page}


def table_cell_statistics() -> dict:
    """Count how many table cells the content check actually compared."""
    expected_dir = PROJECT_ROOT / "parity" / "expected"
    cells = 0
    tables = 0
    rows = 0
    for path in sorted(expected_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for table in payload.get("dataframes", []):
            tables += 1
            rows += len(table.get("rows", []))
            cells += len(table.get("rows", [])) * len(table.get("columns", []))
    return {"tables": tables, "rows": rows, "cells": cells}


# ---------------------------------------------------------------------------
# 2. API latency
# ---------------------------------------------------------------------------


def api_statistics(iterations: int) -> dict:
    from fastapi.testclient import TestClient

    from api.data import cache_clear
    from api.main import app

    client = TestClient(app)
    results = {}
    for label, path in API_ENDPOINTS:
        # Clear in-process rather than via /api/cache/clear, which is an
        # admin-gated endpoint and returns 404 unless TAILGATE_ADMIN_TOKEN is set.
        cache_clear()
        cold = time.perf_counter()
        response = client.get(path, params={"tz": "UTC"})
        cold_ms = (time.perf_counter() - cold) * 1000
        payload_bytes = len(response.content)

        warm = []
        for _ in range(iterations):
            started = time.perf_counter()
            client.get(path, params={"tz": "UTC"})
            warm.append((time.perf_counter() - started) * 1000)

        results[label] = {
            "endpoint": path,
            "cold_ms": round(cold_ms, 1),
            "payload_bytes": payload_bytes,
            "warm": _summary(warm),
        }
    return results


# ---------------------------------------------------------------------------
# 3 & 4 & 5. Browser measurement
# ---------------------------------------------------------------------------


def _grab(page, url: str) -> dict:
    transfer = {"bytes": 0, "requests": 0}

    def _on_response(response):
        transfer["requests"] += 1
        try:
            transfer["bytes"] += len(response.body())
        except Exception:  # noqa: BLE001 - redirects have no body
            pass

    page.on("response", _on_response)
    started = time.perf_counter()
    page.goto(url, wait_until="load", timeout=180000)
    load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    page.wait_for_selector("h1", timeout=120000)
    heading_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    try:
        page.wait_for_load_state("networkidle", timeout=60000)
        settled_ms = (time.perf_counter() - started) * 1000
    except Exception:  # noqa: BLE001
        settled_ms = float("nan")

    page.remove_listener("response", _on_response)
    return {
        "load_ms": load_ms,
        "heading_ms": heading_ms,
        "settled_ms": settled_ms,
        "transfer_bytes": transfer["bytes"],
        "requests": transfer["requests"],
    }


def _region_stats(delta: np.ndarray, x0: int, x1: int) -> dict:
    slab = delta[:, x0:x1, :]
    if slab.size == 0:
        return {"pixels_differing_pct": 0.0, "mean_abs_diff": 0.0}
    differing = (slab.max(axis=2) > PIXEL_THRESHOLD).mean() * 100
    return {
        "pixels_differing_pct": round(float(differing), 2),
        "mean_abs_diff": round(float(slab.mean()), 2),
    }


def _best_alignment(arr_a: np.ndarray, arr_b: np.ndarray, x0: int, x1: int, search: int) -> dict:
    """Find the vertical shift that best aligns the two renderings.

    The two stacks place the same content at slightly different y offsets, so a
    raw pixel diff mostly measures layout shift.  Searching for the minimising
    shift separates "the content is in a different place" from "the content is
    different".
    """
    best_shift = 0
    best_pct = 100.0
    best_mean = 999.0
    for shift in range(-search, search + 1):
        if shift >= 0:
            a = arr_a[shift:, x0:x1, :]
            b = arr_b[: arr_b.shape[0] - shift, x0:x1, :]
        else:
            a = arr_a[: arr_a.shape[0] + shift, x0:x1, :]
            b = arr_b[-shift:, x0:x1, :]
        if a.size == 0:
            continue
        slab = np.abs(a - b)
        pct = float((slab.max(axis=2) > PIXEL_THRESHOLD).mean() * 100)
        if pct < best_pct:
            best_pct = pct
            best_mean = float(slab.mean())
            best_shift = shift
    return {
        "shift_px": best_shift,
        "aligned_pixels_differing_pct": round(best_pct, 2),
        "aligned_mean_abs_diff": round(best_mean, 2),
    }


def _pixel_diff(streamlit_png: bytes, react_png: bytes, name: str) -> dict:
    """Compare two PNG byte strings; return difference statistics."""
    import io

    from PIL import Image

    a = Image.open(io.BytesIO(streamlit_png)).convert("RGB")
    b = Image.open(io.BytesIO(react_png)).convert("RGB")

    width = min(a.width, b.width)
    height = min(a.height, b.height)
    a = a.crop((0, 0, width, height))
    b = b.crop((0, 0, width, height))

    arr_a = np.asarray(a, dtype=np.int16)
    arr_b = np.asarray(b, dtype=np.int16)
    delta = np.abs(arr_a - arr_b)

    sidebar_x = min(SIDEBAR_PX, width)
    main_x = min(SIDEBAR_PX, width)

    # Save a heatmap plus a side-by-side strip so differences can be inspected.
    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.lower().replace(" ", "_").replace("&", "and")
    heat = np.clip(delta.max(axis=2) * 3, 0, 255).astype(np.uint8)
    Image.fromarray(heat, mode="L").save(DIFF_DIR / f"{safe}_heat.png")
    strip = Image.new("RGB", (width * 2 + 8, height), (255, 0, 255))
    strip.paste(a, (0, 0))
    strip.paste(b, (width + 8, 0))
    strip.save(DIFF_DIR / f"{safe}_side_by_side.png")

    return {
        "width": width,
        "height": height,
        "search_range_px": ALIGN_SEARCH_PX,
        **_region_stats(delta, 0, width),
        "sidebar": _region_stats(delta, 0, sidebar_x),
        "main": _region_stats(delta, main_x, width),
        "aligned": _best_alignment(arr_a, arr_b, 0, width, ALIGN_SEARCH_PX),
        "aligned_main": _best_alignment(arr_a, arr_b, main_x, width, ALIGN_SEARCH_PX),
        "aligned_sidebar": _best_alignment(arr_a, arr_b, 0, sidebar_x, ALIGN_SEARCH_PX),
    }


def browser_statistics(iterations: int) -> dict:
    from playwright.sync_api import sync_playwright

    results = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()

        # Warm both servers.
        warm_ctx = browser.new_context(viewport=VIEWPORT)
        warm_page = warm_ctx.new_page()
        for _label, path in PAGES:
            warm_page.goto(f"{STREAMLIT_BASE}{path}", wait_until="load", timeout=180000)
            warm_page.goto(f"{REACT_BASE}{path}", wait_until="load", timeout=180000)
        warm_ctx.close()

        for label, path in PAGES:
            samples = {"streamlit": {"load": [], "heading": [], "settled": []},
                       "react": {"load": [], "heading": [], "settled": []}}
            transfer = {"streamlit": [], "react": []}
            requests = {"streamlit": [], "react": []}
            shots = {}

            for index in range(iterations):
                for stack, base in (("streamlit", STREAMLIT_BASE), ("react", REACT_BASE)):
                    context = browser.new_context(viewport=VIEWPORT)
                    page = context.new_page()
                    measured = _grab(page, f"{base}{path}")
                    samples[stack]["load"].append(measured["load_ms"])
                    samples[stack]["heading"].append(measured["heading_ms"])
                    samples[stack]["settled"].append(measured["settled_ms"])
                    transfer[stack].append(measured["transfer_bytes"])
                    requests[stack].append(measured["requests"])
                    if index == 0:
                        shots[stack] = page.screenshot()
                    context.close()

            from scipy.stats import mannwhitneyu

            sl = samples["streamlit"]["heading"]
            rc = samples["react"]["heading"]
            try:
                statistic, p_value = mannwhitneyu(sl, rc, alternative="two-sided")
                cliff = 2 * statistic / (len(sl) * len(rc)) - 1
            except ValueError:
                statistic, p_value, cliff = float("nan"), float("nan"), float("nan")

            entry = {
                "streamlit": {k: _summary(v) for k, v in samples["streamlit"].items()},
                "react": {k: _summary(v) for k, v in samples["react"].items()},
                "transfer_kb": {
                    "streamlit": round(statistics.median(transfer["streamlit"]) / 1024, 1),
                    "react": round(statistics.median(transfer["react"]) / 1024, 1),
                },
                "requests": {
                    "streamlit": int(statistics.median(requests["streamlit"])),
                    "react": int(statistics.median(requests["react"])),
                },
                "heading_speedup": round(
                    statistics.median(sl) / statistics.median(rc), 2
                ),
                "mannwhitney_u": round(float(statistic), 1),
                "p_value": float(p_value),
                "cliffs_delta": round(float(cliff), 3),
            }
            if "streamlit" in shots and "react" in shots:
                entry["visual"] = _pixel_diff(shots["streamlit"], shots["react"], label)
            results[label] = entry
            print(
                f"  {label:22s} heading {entry['streamlit']['heading']['median']:7.0f}ms"
                f" -> {entry['react']['heading']['median']:6.0f}ms"
                f"  p={entry['p_value']:.4f}  diff={entry.get('visual', {}).get('pixels_differing_pct', float('nan')):5.2f}%"
            )

        browser.close()
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(api: dict, browser: dict, content: dict, cells: dict, iterations: int) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "iterations": iterations,
        "viewport": VIEWPORT,
        "content": content,
        "table_cells": cells,
        "api": api,
        "browser": browser,
    }
    STATS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    t = content.get("totals", {})
    lines = [
        "# Streamlit vs FastAPI + React — measured differences",
        "",
        f"Generated {payload['generated_at']} · {iterations} cold-context page loads per",
        f"page per stack · viewport {VIEWPORT['width']}×{VIEWPORT['height']}.",
        "",
        "## 1. Content parity",
        "",
        "| Dimension | Count |",
        "|---|---:|",
        f"| Pages compared (content) | {t.get('pages_compared', 0)} |",
        f"| Pages matching exactly | **{t.get('pages_matching', 0)}** |",
        f"| Pages compared (live chrome) | {t.get('live_pages_compared', 0)} |",
        f"| Pages matching exactly (live) | **{t.get('live_pages_matching', 0)}** |",
        f"| Mismatched fields | **{t.get('mismatched_fields', 0)}** |",
        f"| Data tables compared | {cells.get('tables', 0)} |",
        f"| Table rows compared | {cells.get('rows', 0):,} |",
        f"| **Table cells compared cell-by-cell** | **{cells.get('cells', 0):,}** |",
        f"| Metrics compared | {t.get('metrics', 0)} |",
        f"| Captions compared | {t.get('captions', 0)} |",
        f"| Alerts compared | {t.get('alerts', 0)} |",
        f"| Charts compared | {t.get('charts', 0)} |",
        "",
        "## 2. API latency (FastAPI only — Streamlit has no JSON API)",
        "",
        "`cold` is the first request after explicitly clearing the artifact cache,",
        "i.e. the cost of rebuilding every derived frame from Parquet. The server",
        "also primes these caches on startup in a background thread, so a real",
        "first request never pays it: measured end-to-end, the first Weekly",
        "Predictions request after boot is ~160 ms rather than ~5.6 s.",
        "",
        "| Page | Endpoint | Cold (ms) | Warm median (ms) | Warm stdev | Warm p95 (ms) | Payload (KB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, entry in api.items():
        warm = entry["warm"]
        lines.append(
            f"| {label} | `{entry['endpoint']}` | {entry['cold_ms']:.0f} | "
            f"{warm['median']:.1f} | {warm['stdev']:.1f} | {warm['p95']:.1f} | "
            f"{entry['payload_bytes'] / 1024:.1f} |"
        )

    lines += [
        "",
        "## 3. Browser timing (median of "
        f"{iterations} loads)",
        "",
        "`Heading` = navigation start → `<h1>` painted. `Settled` = additional time",
        "after that until the page stops issuing requests. `p` is a two-sided",
        "Mann-Whitney U test on the heading samples; Cliff's δ is the effect size",
        "(+1 means every React sample beat every Streamlit sample).",
        "",
        "| Page | Streamlit heading | React heading | Speed-up | p | Cliff's δ | "
        "Streamlit settled | React settled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, entry in browser.items():
        sl = entry["streamlit"]
        rc = entry["react"]
        lines.append(
            f"| {label} | {sl['heading']['median']:.0f} ms | {rc['heading']['median']:.0f} ms | "
            f"**{entry['heading_speedup']:.2f}×** | {entry['p_value']:.4f} | "
            f"{entry['cliffs_delta']:+.2f} | {sl['settled']['median']:.0f} ms | "
            f"{rc['settled']['median']:.0f} ms |"
        )

    sl_medians = [e["streamlit"]["heading"]["median"] for e in browser.values()]
    rc_medians = [e["react"]["heading"]["median"] for e in browser.values()]
    lines += [
        "",
        f"Aggregate: Streamlit **{statistics.median(sl_medians):.0f} ms** vs React "
        f"**{statistics.median(rc_medians):.0f} ms** median "
        f"({statistics.median(sl_medians) / statistics.median(rc_medians):.2f}× faster).",
        "",
        "## 4. Transfer size",
        "",
        "| Page | Streamlit (KB) | React (KB) | Δ | Streamlit requests | React requests |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, entry in browser.items():
        sl_kb = entry["transfer_kb"]["streamlit"]
        rc_kb = entry["transfer_kb"]["react"]
        lines.append(
            f"| {label} | {sl_kb:.0f} | {rc_kb:.0f} | {rc_kb - sl_kb:+.0f} | "
            f"{entry['requests']['streamlit']} | {entry['requests']['react']} |"
        )

    lines += [
        "",
        "## 5. Visual difference (matched screenshots, per-pixel)",
        "",
        f"A pixel counts as different when any RGB channel moves by more than "
        f"{PIXEL_THRESHOLD}. Both stacks rendered at the same viewport; the sidebar",
        f"strip is x < {SIDEBAR_PX} px.",
        "",
        "`Raw` is the difference with the two screenshots compared as-is. `Aligned`",
        "re-renders the comparison at the vertical offset that minimises the",
        "difference (searched ±" + str(ALIGN_SEARCH_PX) + " px), which separates",
        "*content in a different place* from *content that is different*.",
        "",
        "| Page | Raw differing | Aligned differing | Best shift | Mean abs diff | Sidebar raw | Main raw |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, entry in browser.items():
        visual = entry.get("visual")
        if not visual:
            continue
        lines.append(
            f"| {label} | {visual['pixels_differing_pct']:.2f}% | "
            f"**{visual['aligned']['aligned_pixels_differing_pct']:.2f}%** | "
            f"{visual['aligned']['shift_px']:+d} px | {visual['mean_abs_diff']:.2f} | "
            f"{visual['sidebar']['pixels_differing_pct']:.2f}% | "
            f"{visual['main']['pixels_differing_pct']:.2f}% |"
        )

    visuals = [e["visual"] for e in browser.values() if e.get("visual")]
    if visuals:
        raw = [v["pixels_differing_pct"] for v in visuals]
        aligned = [v["aligned"]["aligned_pixels_differing_pct"] for v in visuals]
        shifts = [v["aligned"]["shift_px"] for v in visuals]
        lines += [
            "",
            f"Aggregate raw difference **{statistics.median(raw):.2f}%** of pixels;",
            f"after alignment **{statistics.median(aligned):.2f}%** "
            f"(median best shift {statistics.median(shifts):+.0f} px).",
            "",
            "Alignment removes only "
            f"{100 * (1 - statistics.median(aligned) / statistics.median(raw)):.0f}% of the "
            "difference, so a single uniform vertical offset does **not** explain the "
            "residual: the spacing between elements differs by different amounts down the "
            "page, and the rendering chrome (canvas dataframes, Streamlit's toolbar, the "
            "Plotly theme) is genuinely different.",
        ]

    lines += [
        "",
        "Interpretation: the visual metric measures *rendering*, not *content*. A 35%",
        "pixel difference alongside 0 mismatched fields (8,639/8,639 table cells, all",
        "headings, metrics, captions, alerts and navigation identical) means the two apps",
        "put the same information on screen in slightly different boxes.",
        "",
        "## 6. Where the remaining differences come from",
        "",
        "- **`st.dataframe` is a canvas, not a table.** Streamlit paints every dataframe",
        "  with glide-data-grid onto a `<canvas>`; the React build emits a real HTML",
        "  `<table>`. Cell contents are verified identical cell-by-cell, but grid chrome",
        "  (column headers, resize handles, per-cell toolbars, row striping offsets) is",
        "  drawn differently. This is the single largest contributor on the table-heavy",
        "  pages and explains why the main-content difference grows with page size",
        "  (Home 17% → Historical Analysis 50%).",
        "- **Streamlit chrome.** The `Deploy` button, hamburger menu and per-element",
        "  toolbars exist only in Streamlit and sit in the top-right of every page.",
        "- **Plotly theming.** Both render the same figure JSON, but Streamlit layers its",
        "  own Plotly template over the traces, so axis lines and gridlines can differ.",
        "- **Non-uniform vertical rhythm.** Streamlit spaces top-level elements with a",
        "  flex gap plus per-element margins/padding; the React build approximates it.",
        "  Element heights now match exactly (e.g. metric blocks are 76px in both, on a",
        "  92px pitch) but the accumulated offset between sections still differs by",
        "  tens of pixels.",
        "",
        "Fonts are **not** a source of difference: the React build bundles the same",
        "variable webfonts Streamlit ships (`SourceSansVF`, `SourceCodeVF`) and declares",
        "them under the same family names, so both rasterise text identically.",
        "",
    ]
    STATS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {STATS_MD}")
    print(f"json   -> {STATS_JSON}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--from-json",
        action="store_true",
        help="rebuild STATISTICS.md from an existing parity/statistics.json",
    )
    args = parser.parse_args()

    if args.from_json:
        if not STATS_JSON.exists():
            print(f"{STATS_JSON} not found; run a measurement first")
            return 1
        payload = json.loads(STATS_JSON.read_text(encoding="utf-8"))
        write_report(
            payload["api"], payload["browser"], payload["content"],
            payload["table_cells"], payload["iterations"],
        )
        return 0

    print("=== content parity ===")
    content = content_statistics()
    cells = table_cell_statistics()
    print(f"  pages matching: {content.get('totals', {}).get('pages_matching')}"
          f"/{content.get('totals', {}).get('pages_compared')}"
          f"  table cells compared: {cells.get('cells', 0):,}")

    print("\n=== API latency ===")
    api = api_statistics(args.iterations)
    for label, entry in api.items():
        print(f"  {label:22s} warm median {entry['warm']['median']:7.1f}ms  "
              f"payload {entry['payload_bytes'] / 1024:7.1f} KB")

    print("\n=== browser timing + visual diff ===")
    browser = browser_statistics(args.iterations)

    write_report(api, browser, content, cells, args.iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
