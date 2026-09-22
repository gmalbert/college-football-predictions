"""Break down where the Weekly Predictions time actually goes.

Three layers are measured separately:

1. Server-side: which part of ``build_weekly`` burns the time (stage timings).
2. HTTP: how long the JSON request itself takes, warm and cold.
3. Browser: request time vs. time from "JSON arrived" to "<h1> painted" vs.
   "table fully rendered" — i.e. server cost versus DOM cost.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEEKLY = "/api/weekly-predictions"
OTHER = "/api/model-performance"


def stage_profile(iterations: int = 5) -> None:
    """Time each stage of build_weekly by patching the module's helpers."""
    import api.services.weekly as W

    from utils import betting, odds_ingestion
    from utils.models import predict_for_display

    timings: dict[str, list[float]] = {}

    def wrap(module, name, label):
        original = getattr(module, name)

        def timed(*args, **kwargs):
            started = time.perf_counter()
            result = original(*args, **kwargs)
            timings.setdefault(label, []).append((time.perf_counter() - started) * 1000)
            return result

        setattr(module, name, timed)

    wrap(W, "load_feature_matrix", "load feature_matrix (cached)")
    wrap(W, "predict_for_display", "predict_for_display")
    wrap(W, "load_market_snapshots", "load line_snapshots (cached)")
    wrap(W, "build_market_consensus_from_snapshots", "build_market_consensus_from_snapshots")
    wrap(betting, "generate_spread_pick", "generate_spread_pick")
    wrap(betting, "generate_total_pick", "generate_total_pick")

    # Prime the caches once, then measure.
    W.build_weekly(timezone_name="UTC")
    for _ in range(iterations):
        timings.clear()
        started = time.perf_counter()
        payload = W.build_weekly(timezone_name="UTC")
        total = (time.perf_counter() - started) * 1000
        print(f"  total build_weekly: {total:8.1f} ms   rows={len(payload['table']['rows'])}")
        for label, samples in timings.items():
            count = len(samples)
            print(f"      {label:<42} {sum(samples):8.1f} ms  x{count:<4} "
                  f"({sum(samples) / count:.3f} ms each)")


def http_profile(iterations: int = 12) -> None:
    import json
    import urllib.request

    def fetch(path: str) -> tuple[float, int]:
        started = time.perf_counter()
        with urllib.request.urlopen(f"http://127.0.0.1:8000{path}", timeout=120) as response:
            body = response.read()
        return (time.perf_counter() - started) * 1000, len(body)

    for label, path in (("weekly", WEEKLY), ("model-performance", OTHER)):
        cold, size = fetch(path)
        warm = [fetch(path)[0] for _ in range(iterations)]
        print(f"  {label:<18} cold={cold:7.1f} ms  warm median={statistics.median(warm):7.1f} ms"
              f"  min={min(warm):7.1f}  max={max(warm):7.1f}  payload={size / 1024:.1f} KB")


def browser_profile(iterations: int = 5) -> None:
    from playwright.sync_api import sync_playwright

    js = r"""
    () => {
      const nav = performance.getEntriesByType('navigation')[0];
      const api = performance.getEntriesByType('resource')
        .filter(r => r.name.includes('/api/'));
      const h1 = document.querySelector('h1');
      const rows = document.querySelectorAll('table.df tbody tr');
      const cells = document.querySelectorAll('table.df tbody td');
      return {
        domContentLoaded: Math.round(nav.domContentLoadedEventEnd),
        loadEvent: Math.round(nav.loadEventEnd),
        apiMs: api.map(r => ({ name: r.name.split('/api/')[1].split('?')[0],
                               ms: Math.round(r.duration), size: r.transferSize })),
        apiTotalMs: Math.round(api.reduce((a, r) => a + r.duration, 0)),
        h1Present: !!h1,
        rowCount: rows.length,
        cellCount: cells.length,
        domNodes: document.getElementsByTagName('*').length,
      };
    }
    """

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for stack, base, path in (
            ("streamlit", "http://127.0.0.1:8501", "/Weekly_Predictions"),
            ("react", "http://127.0.0.1:8000", "/Weekly_Predictions"),
        ):
            heading: list[float] = []
            details = []
            for _ in range(iterations):
                context = browser.new_context(viewport={"width": 1440, "height": 900})
                page = context.new_page()
                started = time.perf_counter()
                page.goto(f"{base}{path}", wait_until="load", timeout=180000)
                page.wait_for_selector("h1", timeout=120000)
                heading.append((time.perf_counter() - started) * 1000)
                details.append(page.evaluate(js))
                context.close()
            print(f"  {stack:<10} heading median={statistics.median(heading):7.0f} ms")
            sample = details[-1]
            print(f"             domContentLoaded={sample['domContentLoaded']} ms"
                  f"  load={sample['loadEvent']} ms"
                  f"  api_total={sample['apiTotalMs']} ms")
            for entry in sample["apiMs"]:
                print(f"             api {entry['name']:<24} {entry['ms']:>5} ms  "
                      f"{entry['size']:>8} bytes")
            print(f"             rows={sample['rowCount']}  cells={sample['cellCount']}"
                  f"  DOM nodes={sample['domNodes']}")
        browser.close()


def main() -> int:
    print("=== 1. server-side stage profile (in-process) ===")
    stage_profile()
    print("\n=== 2. HTTP latency against the live server ===")
    http_profile()
    print("\n=== 3. browser breakdown ===")
    browser_profile()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
