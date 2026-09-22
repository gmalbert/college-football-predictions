/**
 * Plotly renderer.
 *
 * The backend serialises each figure with ``api.charts.figure_json`` using the
 * same trace/layout definitions the Streamlit pages build, so both apps draw
 * identical charts from identical JSON.
 */
import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

const CONFIG = {
  responsive: true,
  displaylogo: false,
  scrollZoom: true,
};

export function PlotlyChart({ figure, height, testId = "stPlotlyChart" }) {
  const ref = useRef(null);

  useEffect(() => {
    const node = ref.current;
    if (!node || !figure) return undefined;
    const layout = height ? { ...figure.layout, height } : figure.layout;
    Plotly.react(node, figure.data, layout, CONFIG);

    // Tab panels start hidden (matching Streamlit), so a chart created at zero
    // size has to be resized once its panel becomes visible.
    const observer =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => {
            if (node.clientWidth > 0 && node.clientHeight > 0) {
              Plotly.Plots.resize(node);
            }
          })
        : null;
    if (observer) observer.observe(node);

    return () => {
      if (observer) observer.disconnect();
      Plotly.purge(node);
    };
  }, [figure, height]);

  if (!figure) return null;

  return (
    <div
      className="stPlotlyChart"
      data-testid={testId}
      data-plotly-title={figure.layout?.title?.text || figure.layout?.title || ""}
    >
      <div ref={ref} />
    </div>
  );
}

export default PlotlyChart;
