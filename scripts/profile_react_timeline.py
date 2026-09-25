"""Timeline breakdown of the React Weekly Predictions load."""
from __future__ import annotations

import json
import statistics
import time

from playwright.sync_api import sync_playwright

JS = r"""
() => {
  const nav = performance.getEntriesByType('navigation')[0];
  const paints = {};
  performance.getEntriesByType('paint').forEach(p => { paints[p.name] = Math.round(p.startTime); });
  const res = performance.getEntriesByType('resource').map(r => ({
    name: r.name.replace(location.origin, ''),
    start: Math.round(r.startTime),
    dur: Math.round(r.duration),
    size: r.transferSize,
    type: r.initiatorType,
  }));
  return {
    navigationStart: 0,
    domInteractive: Math.round(nav.domInteractive),
    domContentLoaded: Math.round(nav.domContentLoadedEventEnd),
    loadEvent: Math.round(nav.loadEventEnd),
    fcp: paints['first-contentful-paint'] ?? null,
    fp: paints['first-paint'] ?? null,
    resources: res,
    domNodes: document.getElementsByTagName('*').length,
  };
}
"""

with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    samples = []
    for i in range(4):
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        started = time.perf_counter()
        page.goto("http://127.0.0.1:8000/Weekly_Predictions", wait_until="load", timeout=180000)
        page.wait_for_selector("h1", timeout=120000)
        heading = (time.perf_counter() - started) * 1000
        data = page.evaluate(JS)
        data["heading_after_load_ms"] = round(heading - data["loadEvent"], 1)
        samples.append(data)
        context.close()
    browser.close()

for i, s in enumerate(samples):
    print(f"--- run {i} ---")
    print(f"  domInteractive      {s['domInteractive']:>6} ms")
    print(f"  first-paint         {s['fp']}")
    print(f"  first-contentful    {s['fcp']}")
    print(f"  domContentLoaded    {s['domContentLoaded']:>6} ms")
    print(f"  loadEvent           {s['loadEvent']:>6} ms")
    print(f"  DOM nodes           {s['domNodes']:>6}")
    biggest = sorted(s["resources"], key=lambda r: -r["dur"])[:6]
    for r in biggest:
        print(f"    {r['type']:<10} start={r['start']:>5} dur={r['dur']:>5} "
              f"{r['size']:>9} B  {r['name'][:60]}")

print("\n=== medians ===")
for key in ("domInteractive", "fcp", "domContentLoaded", "loadEvent", "domNodes"):
    values = [s[key] for s in samples if s[key] is not None]
    if values:
        print(f"  {key:<20} {statistics.median(values):>8.0f}")
api = [r for s in samples for r in s["resources"] if "/api/" in r["name"]]
for name in sorted({r["name"].split("?")[0] for r in api}):
    durs = [r["dur"] for r in api if r["name"].startswith(name)]
    print(f"  api {name:<28} median {statistics.median(durs):>7.0f} ms  (n={len(durs)})")
