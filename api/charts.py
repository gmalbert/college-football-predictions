"""Plotly figure serialisation with a Streamlit-compatible default template.

Streamlit renders Plotly figures through its own theme.  The page code then
overrides ``paper_bgcolor`` / ``plot_bgcolor`` / ``font.color`` explicitly, so
those values win.  What is left for the theme to control is the trace colourway
and the grid styling — replicated here so the React charts match.
"""
from __future__ import annotations

import json
from typing import Any

import plotly.graph_objects as go

# Streamlit's Plotly light-theme colourway.
STREAMLIT_COLORWAY = [
    "#83C9FF", "#0068C9", "#FF2B2B", "#FFABAB", "#29B09D",
    "#7DEFA1", "#FF8700", "#FFD16D", "#6D3FC0", "#D5DAE5",
]

STREAMLIT_TEMPLATE: dict[str, Any] = {
    "layout": {
        "colorway": STREAMLIT_COLORWAY,
        "font": {"color": "#31333F", "size": 14},
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#F0F2F6",
        "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
        "xaxis": {
            "gridcolor": "#E6EAF1",
            "linecolor": "#E6EAF1",
            "zerolinecolor": "#E6EAF1",
            "automargin": True,
        },
        "yaxis": {
            "gridcolor": "#E6EAF1",
            "linecolor": "#E6EAF1",
            "zerolinecolor": "#E6EAF1",
            "automargin": True,
        },
    }
}


def figure_json(fig: go.Figure) -> dict:
    """Serialise a Plotly figure to a JSON-safe ``{data, layout}`` payload."""
    fig.update_layout(template=STREAMLIT_TEMPLATE)
    payload = json.loads(fig.to_json())
    payload.pop("frames", None)
    return {"data": payload.get("data", []), "layout": payload.get("layout", {})}
