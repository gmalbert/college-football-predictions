/**
 * Lazy wrapper around the Plotly renderer.
 *
 * ``plotly.js`` is ~4.8 MB and lived in the main bundle, so *every* route paid
 * for it at startup — including the four pages that render no charts at all
 * (Home, Weekly Predictions, Value Bets, Data & Model Quality).  Importing the
 * real component lazily puts Plotly in its own chunk that Vite only fetches
 * when a chart actually mounts.
 */
import { Suspense, lazy } from "react";

const LazyPlotly = lazy(() => import("./PlotlyChart"));

export function PlotlyChart(props) {
  return (
    <Suspense
      fallback={<div className="stPlotlyChart" style={{ minHeight: "120px" }} />}
    >
      <LazyPlotly {...props} />
    </Suspense>
  );
}

export default PlotlyChart;
