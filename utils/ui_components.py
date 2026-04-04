"""utils/ui_components.py — Reusable Streamlit UI components."""
from __future__ import annotations
import streamlit as st


def render_sidebar() -> None:
    """Standard sidebar rendered on every page."""
    with st.sidebar:
        from pathlib import Path
        logo = Path(__file__).resolve().parent.parent / "data_files" / "logo.png"
        if logo.exists():
            st.image(str(logo), width=200)

        st.divider()

        # Live model metrics (if available)
        try:
            from utils.models import load_metrics, models_trained
            if models_trained():
                m = load_metrics()
                ats = m.get("ats", {})
                win = m.get("win_model", {})
                if ats:
                    wins   = ats.get("wins", 0)
                    losses = ats.get("losses", 0)
                    pct    = ats.get("pct", 0)
                    st.metric("Season ATS", f"{wins}‑{losses}", f"{pct:.1%} win rate")
                if win.get("brier"):
                    st.metric("Model Brier", f"{win['brier']:.4f}", help="Lower is better")
        except Exception:
            pass

        st.divider()
        st.caption("Data: CFBD API · ESPN | Updated daily")
        st.divider()


def game_card(
    home: str,
    away: str,
    model_spread: float,
    book_spread: float,
    win_prob: float,
    model_total: float | None = None,
    book_total: float | None = None,
) -> None:
    """Render a single game prediction card."""
    edge = abs(model_spread - book_spread)
    edge_color = "🟢" if edge >= 2 else "🟡" if edge >= 1 else "⚪"

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"### {away}")
        with col2:
            st.markdown("### @")
        with col3:
            st.markdown(f"### {home}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Prob", f"{win_prob:.0%}")
        c2.metric("Model Spread", f"{model_spread:+.1f}")
        c3.metric("Book Spread", f"{book_spread:+.1f}")
        c4.metric("Edge", f"{edge:.1f} {edge_color}")

        if model_total and book_total:
            t1, t2 = st.columns(2)
            t1.metric("Model Total", f"{model_total:.1f}")
            t2.metric("Book Total", f"{book_total:.1f}")

        st.divider()


def value_badge(edge: float) -> str:
    """Return a colored badge string based on edge size."""
    if edge >= 3:
        return "🔥 STRONG VALUE"
    elif edge >= 2:
        return "✅ VALUE"
    elif edge >= 1:
        return "⚠️ LEAN"
    return ""


def metric_row(metrics: dict[str, tuple[str, str]]) -> None:
    """
    Render a row of st.metric() calls.
    metrics: {"Label": ("value", "delta"), ...}
    """
    cols = st.columns(len(metrics))
    for col, (label, (value, delta)) in zip(cols, metrics.items()):
        col.metric(label, value, delta)
