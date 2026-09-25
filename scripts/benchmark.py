"""Speed and payload benchmark: Streamlit vs FastAPI + React.

Three measurements are collected:

1. **API latency** — in-process timings for every FastAPI endpoint, cold
   (cache cleared) versus warm, reported as min/median/p95/max plus payload size.
2. **Page load** — Playwright navigation timings against the live Streamlit
   server and the live React build: time to ``load``, time until the ``<h1>``
   is painted, and time until the page stops making requests.
3. **Payload transfer** — bytes the browser actually receives for a page load.

Usage::

    python scripts/benchmark.py --iterations 7
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_JSON = PROJECT_ROOT / "parity" / "benchmark_report.json"
REPORT_MD = PROJECT_ROOT / "parity" / "BENCHMARK.md"

STREAMLIT_BASE = "http://127.0.0.1:8501"
REACT_BASE = "http://127.0.0.1:8000"

# (label, page path, api path, api params)
PAGES = [
    ("Home", "/", "/api/home", {}),
    ("Weekly Predictions", "/Weekly_Predictions", "/api/weekly-predictions", {"tz": "UTC"}),
    ("Value Bets", "/Value_Bets", "/api/value-bets", {}),
    ("Team Explorer", "/Team_Explorer", "/api/team-explorer", {}),
    ("Historical Analysis", "/Historical_Analysis", "/api/historical-analysis", {}),
    ("Model Performance", "/Model_Performance", "/api/model-performance", {}),
    ("Win Probability", "/Win_Probability", "/api/win-probability", {}),
    ("Preseason Outlook", "/Preseason_Outlook", "/api/preseason-outlook", {}),
    ("Data & Model Quality", "/Data_Quality", "/api/data-quality", {}),
]


def summarise(samples: list[float]) -> dict:
    ordered = sorted(samples)
    if not ordered:
        return {}
    index_95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "n": len(ordered),
        "min_ms": round(ordered[0], 2),
        "median_ms": round(statistics.median(ordered), 2),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "p95_ms": round(ordered[index_95], 2),
        "max_ms": round(ordered[-1], 2),
        "stdev_ms": round(statistics.pstdev(ordered), 2) if len(ordered) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# 1. API latency
# ---------------------------------------------------------------------------


def benchmark_api(iterations: int) -> dict:
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    results: dict[str, dict] = {}

    for label, _path, api_path, params in PAGES:
        client.post("/api/cache/clear")
        cold_start = time.perf_counter()
        cold_response = client.get(api_path, params=params)
        cold_ms = (time.perf_counter() - cold_start) * 1000
        size = len(cold_response.content)

        warm: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            response = client.get(api_path, params=params)
            warm.append((time.perf_counter() - started) * 1000)
            assert response.status_code == 200, f"{api_path} -> {response.status_code}"

        results[label] = {
            "endpoint": api_path,
            "cold_ms": round(cold_ms, 2),
            "payload_bytes": size,
            "payload_kb": round(size / 1024, 1),
            "warm": summarise(warm),
        }
        print(
            f"  {label:22s} cold={cold_ms:8.1f}ms  warm_median={results[label]['warm']['median_ms']:6.2f}ms"
            f"  payload={size / 1024:7.1f} KB"
        )
    return results


# ---------------------------------------------------------------------------
# 2 & 3. Browser page load
# ---------------------------------------------------------------------------


def _measure_page(page, url: str, wait_for_selector: str) -> dict:
    transfer = {"bytes": 0, "requests": 0}

    def _on_response(response):
        try:
            body = response.body()
            transfer["bytes"] += len(body)
            transfer["requests"] += 1
        except Exception:  # noqa: BLE001 - redirects/preflights have no body
            transfer["requests"] += 1

    page.on("response", _on_response)
    started = time.perf_counter()
    page.goto(url, wait_until="load", timeout=180000)
    load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    try:
        page.wait_for_selector(wait_for_selector, timeout=120000)
        heading_ms = (time.perf_counter() - started) * 1000
    except Exception:  # noqa: BLE001
        heading_ms = float("nan")

    started = time.perf_counter()
    try:
        page.wait_for_load_state("networkidle", timeout=120000)
        settled_ms = (time.perf_counter() - started) * 1000
    except Exception:  # noqa: BLE001
        settled_ms = float("nan")

    page.remove_listener("response", _on_response)
    return {
        "load_ms": load_ms,
        "heading_ms": heading_ms,
        "settled_ms": settled_ms,
        "transfer_kb": round(transfer["bytes"] / 1024, 1),
        "requests": transfer["requests"],
    }


def benchmark_pages(iterations: int) -> dict:
    from playwright.sync_api import sync_playwright

    results: dict[str, dict] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()

        # Warm both servers first so the first sample is not an outlier.
        warm_context = browser.new_context(viewport={"width": 1600, "height": 1000})
        warm_page = warm_context.new_page()
        for _label, path, _api, _params in PAGES:
            warm_page.goto(f"{STREAMLIT_BASE}{path}", wait_until="load", timeout=180000)
            warm_page.goto(f"{REACT_BASE}{path}", wait_until="load", timeout=180000)
        warm_context.close()

        for label, path, _api, _params in PAGES:
            samples: dict[str, list[float]] = {
                "streamlit_load": [], "streamlit_heading": [], "streamlit_settled": [],
                "react_load": [], "react_heading": [], "react_settled": [],
            }
            transfer = {"streamlit_kb": [], "react_kb": []}
            requests = {"streamlit": [], "react": []}

            for _ in range(iterations):
                for target, base, prefix in (
                    ("streamlit", STREAMLIT_BASE, "streamlit"),
                    ("react", REACT_BASE, "react"),
                ):
                    context = browser.new_context(viewport={"width": 1600, "height": 1000})
                    page = context.new_page()
                    measured = _measure_page(page, f"{base}{path}", "h1")
                    samples[f"{prefix}_load"].append(measured["load_ms"])
                    samples[f"{prefix}_heading"].append(measured["heading_ms"])
                    samples[f"{prefix}_settled"].append(measured["settled_ms"])
                    transfer[f"{prefix}_kb"].append(measured["transfer_kb"])
                    requests[prefix].append(measured["requests"])
                    context.close()

            entry = {
                "streamlit": {
                    "load": summarise(samples["streamlit_load"]),
                    "heading_visible": summarise(samples["streamlit_heading"]),
                    "network_settled": summarise(samples["streamlit_settled"]),
                    "transfer_kb_median": round(statistics.median(transfer["streamlit_kb"]), 1),
                    "requests_median": int(statistics.median(requests["streamlit"])),
                },
                "react": {
                    "load": summarise(samples["react_load"]),
                    "heading_visible": summarise(samples["react_heading"]),
                    "network_settled": summarise(samples["react_settled"]),
                    "transfer_kb_median": round(statistics.median(transfer["react_kb"]), 1),
                    "requests_median": int(statistics.median(requests["react"])),
                },
            }
            speedup = (
                entry["streamlit"]["heading_visible"]["median_ms"]
                / entry["react"]["heading_visible"]["median_ms"]
                if entry["react"]["heading_visible"]["median_ms"]
                else float("nan")
            )
            entry["heading_speedup"] = round(speedup, 2)
            results[label] = entry
            print(
                f"  {label:22s} streamlit={entry['streamlit']['heading_visible']['median_ms']:8.0f}ms"
                f"  react={entry['react']['heading_visible']['median_ms']:7.0f}ms"
                f"  speedup={speedup:5.2f}x"
                f"  payload={entry['streamlit']['transfer_kb_median']:7.1f}->{entry['react']['transfer_kb_median']:6.1f} KB"
            )

        browser.close()
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(api: dict, pages: dict, iterations: int) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "iterations": iterations,
        "api": api,
        "pages": pages,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Parity build — speed and payload benchmark",
        "",
        f"Generated {payload['generated_at']} · {iterations} iterations per measurement.",
        "",
        "## 1. FastAPI endpoint latency",
        "",
        "`cold` is the first request after clearing the artifact cache; `warm` is",
        "the median of the repeated requests that follow.",
        "",
        "| Page | Endpoint | Cold (ms) | Warm median (ms) | Warm p95 (ms) | Payload (KB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, entry in api.items():
        lines.append(
            f"| {label} | `{entry['endpoint']}` | {entry['cold_ms']:.1f} | "
            f"{entry['warm']['median_ms']:.2f} | {entry['warm']['p95_ms']:.2f} | "
            f"{entry['payload_kb']:.1f} |"
        )

    lines += [
        "",
        "## 2. Browser page load (median of "
        f"{iterations} cold-context loads)",
        "",
        "`Heading` is the time from navigation start until the page `<h1>` is",
        "painted — the point at which the page is readable.",
        "",
        "`Settled` is the *additional* time after that until the page stops",
        "issuing network requests, i.e. how much longer the user waits for the",
        "page to finish streaming in. Streamlit re-runs the whole script and",
        "streams elements in; the React build fetches one JSON payload.",
        "",
        "| Page | Streamlit heading (ms) | React heading (ms) | Speed-up | "
        "Streamlit settled (ms) | React settled (ms) | Streamlit KB | React KB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, entry in pages.items():
        lines.append(
            f"| {label} | {entry['streamlit']['heading_visible']['median_ms']:.0f} | "
            f"{entry['react']['heading_visible']['median_ms']:.0f} | "
            f"{entry['heading_speedup']:.2f}× | "
            f"{entry['streamlit']['network_settled']['median_ms']:.0f} | "
            f"{entry['react']['network_settled']['median_ms']:.0f} | "
            f"{entry['streamlit']['transfer_kb_median']:.1f} | "
            f"{entry['react']['transfer_kb_median']:.1f} |"
        )

    streamlit_median = statistics.median(
        entry["streamlit"]["heading_visible"]["median_ms"] for entry in pages.values()
    )
    react_median = statistics.median(
        entry["react"]["heading_visible"]["median_ms"] for entry in pages.values()
    )
    lines += [
        "",
        "## 3. Summary",
        "",
        f"- Median page heading-visible time — Streamlit **{streamlit_median:.0f} ms** "
        f"vs React **{react_median:.0f} ms** "
        f"({streamlit_median / react_median:.2f}× faster).",
        f"- Warm API median across all endpoints — "
        f"**{statistics.median(e['warm']['median_ms'] for e in api.values()):.2f} ms**.",
        f"- Largest JSON payload — "
        f"**{max(e['payload_kb'] for e in api.values()):.1f} KB** "
        f"({max(api, key=lambda k: api[k]['payload_kb'])}).",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {REPORT_MD}")
    print(f"json   -> {REPORT_JSON}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--skip-browser", action="store_true")
    args = parser.parse_args()

    print("=== 1. API endpoint latency ===")
    api = benchmark_api(args.iterations)

    pages: dict = {}
    if not args.skip_browser:
        print("\n=== 2. Browser page load (Streamlit vs React) ===")
        pages = benchmark_pages(args.iterations)

    write_report(api, pages, args.iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
