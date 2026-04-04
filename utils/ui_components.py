"""utils/ui_components.py — Reusable Streamlit UI components."""
from __future__ import annotations
import datetime
import streamlit as st

# ---------------------------------------------------------------------------
# Day / Night themes — auto-selected by server clock (6 am – 10 pm = day)
# ---------------------------------------------------------------------------
THEMES: dict[str, dict] = {
    "day": {
        "background": "#EEF4FB", "sidebar": "#D8E8F5", "card": "#FFFFFF",
        "text": "#1A2B3C", "secondary_text": "#4A6E8A", "accent": "#2B7CB8",
        "border": "#AECDE6", "link": "#1A5F96",
        "header": "#EEF4FB", "nav_active_bg": "#2B7CB8", "nav_active_text": "#FFFFFF",
        "metric_label": "#4A6E8A", "metric_value": "#1A2B3C",
    },
    "night": {
        "background": "#0F172A", "sidebar": "#111827", "card": "#1E293B",
        "text": "#F1F5F9", "secondary_text": "#CBD5E1", "accent": "#F59E0B",
        "border": "#334155", "link": "#FBBF24",
        "header": "#0F172A", "nav_active_bg": "#1E3A5F", "nav_active_text": "#F1F5F9",
        "metric_label": "#CBD5E1", "metric_value": "#F1F5F9",
    },
}


def _auto_theme_name() -> str:
    """Return 'day' between 06:00 and 21:59, else 'night'."""
    hour = datetime.datetime.now().hour
    return "day" if 6 <= hour < 22 else "night"


def apply_theme(name: str | None = None) -> None:
    """Inject CSS for the given theme key ('day'/'night'). Defaults to auto."""
    if name is None:
        name = _auto_theme_name()
    theme = THEMES.get(name, THEMES["day"])
    bg   = theme["background"]
    sb   = theme["sidebar"]
    card = theme["card"]
    tx   = theme["text"]
    stx  = theme["secondary_text"]
    acc  = theme["accent"]
    bdr  = theme["border"]
    lnk  = theme["link"]
    # header bar and nav active colours — fall back gracefully for light themes
    hdr  = theme.get("header", bg)
    nav_bg   = theme.get("nav_active_bg",   acc)
    nav_tx   = theme.get("nav_active_text",  "#FFFFFF")
    mlabel   = theme.get("metric_label",     stx)
    mvalue   = theme.get("metric_value",     tx)

    css = (
        "<style>"

        # ── Full-page background & base text ──────────────────────────────
        "html, body {"
        f"background-color: {bg} !important;"
        f"color: {tx} !important;"
        "background-image: none !important;"
        "}"

        # ── Top header bar (where Deploy button lives) ────────────────────
        "[data-testid=\"stHeader\"], header, [data-testid=\"stToolbar\"] {"
        f"background-color: {hdr} !important;"
        "background-image: none !important;"
        "}"

        # ── Main app view container ───────────────────────────────────────
        "[data-testid=\"stAppViewContainer\"], .main, section.main, .block-container,"
        " body > div, body > div > div, body > div > div > div,"
        " body > div > div > div > section, body > div > div > div > section > div {"
        f"background-color: {bg} !important;"
        f"color: {tx} !important;"
        "background-image: none !important;"
        "}"

        # ── Sidebar — blanket coverage of every child element ────────────
        "[data-testid=\"stSidebar\"],"
        "[data-testid=\"stSidebar\"] > *,"
        "[data-testid=\"stSidebar\"] * {"
        f"background-color: {sb} !important;"
        f"color: {tx} !important;"
        "background-image: none !important;"
        "}"

        # Active nav link and ALL its children
        "[data-testid=\"stSidebarNavLink\"][aria-current=\"page\"],"
        "[data-testid=\"stSidebarNavLink\"][aria-current=\"page\"] * {"
        f"background-color: {nav_bg} !important;"
        f"color: {nav_tx} !important;"
        "border-radius: 6px !important;"
        "}"

        # Sidebar form controls (selectbox etc.) keep card bg for legibility
        "[data-testid=\"stSidebar\"] .stSelectbox > div > div,"
        "[data-testid=\"stSidebar\"] .stSelectbox > div > div > div {"
        f"background-color: {card} !important;"
        f"color: {tx} !important;"
        f"border-color: {bdr} !important;"
        "}"

        # ── Card / inner containers ───────────────────────────────────────
        ".css-1d391kg, .css-18e3th9, .css-1v0mbdj, .css-10trblm, .css-1wrcr25, .css-1lcbmhc {"
        f"background-color: {card} !important;"
        f"color: {tx} !important;"
        f"border-color: {bdr} !important;"
        "}"

        # ── Headings ──────────────────────────────────────────────────────
        "h1, h2, h3, h4, h5, h6 {"
        f"color: {tx} !important;"
        "}"

        # ── Paragraphs / markdown ─────────────────────────────────────────
        "p, .stMarkdown, .stMarkdown p, .stText {"
        f"color: {tx} !important;"
        "}"

        # ── Links ─────────────────────────────────────────────────────────
        "a, a:visited {"
        f"color: {lnk} !important;"
        "}"

        # ── Secondary / caption text ──────────────────────────────────────
        ".css-1n543e5, .css-1bzijpv, .css-16huue1, small, caption {"
        f"color: {stx} !important;"
        "}"

        # ── Metric widgets ────────────────────────────────────────────────
        "[data-testid=\"stMetricLabel\"], [data-testid=\"stMetricLabel\"] * {"
        f"color: {mlabel} !important;"
        "}"
        "[data-testid=\"stMetricValue\"], [data-testid=\"stMetricValue\"] * {"
        f"color: {mvalue} !important;"
        "}"
        "[data-testid=\"stMetricDelta\"], [data-testid=\"stMetricDelta\"] * {"
        f"color: {stx} !important;"
        "}"

        # ── Buttons ───────────────────────────────────────────────────────
        ".stButton>button, button {"
        f"background-color: {acc} !important;"
        "color: #FFFFFF !important;"
        f"border-color: {acc} !important;"
        "}"

        # ── Form inputs ───────────────────────────────────────────────────
        ".stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {"
        f"background-color: {card} !important;"
        f"color: {tx} !important;"
        f"border-color: {bdr} !important;"
        "}"

        # ── Dividers ─────────────────────────────────────────────────────
        "hr {"
        f"border-color: {bdr} !important;"
        "}"

        # ── Tables ────────────────────────────────────────────────────────
        "table, th, td {"
        f"color: {tx} !important;"
        f"border-color: {bdr} !important;"
        "}"

        # ── Tabs ──────────────────────────────────────────────────────────
        # Tab bar container
        "[data-testid=\"stTabs\"] [role=\"tablist\"] {"
        f"background-color: {bg} !important;"
        f"border-bottom: 2px solid {bdr} !important;"
        "gap: 4px !important;"
        "}"
        # Every tab button
        "[data-testid=\"stTabs\"] [role=\"tab\"] {"
        f"background-color: transparent !important;"
        f"color: {stx} !important;"
        "border: none !important;"
        "border-radius: 0 !important;"
        "padding: 8px 18px !important;"
        "font-weight: 500 !important;"
        f"border-bottom: 3px solid transparent !important;"
        "margin-bottom: -2px !important;"
        "transition: color 0.15s, border-color 0.15s !important;"
        "}"
        # Hover
        "[data-testid=\"stTabs\"] [role=\"tab\"]:hover {"
        f"color: {tx} !important;"
        f"border-bottom-color: {bdr} !important;"
        "background-color: transparent !important;"
        "}"
        # Active / selected tab
        "[data-testid=\"stTabs\"] [role=\"tab\"][aria-selected=\"true\"] {"
        f"color: {acc} !important;"
        f"border-bottom: 3px solid {acc} !important;"
        "background-color: transparent !important;"
        "font-weight: 700 !important;"
        "}"
        # Tab panel content area
        "[data-testid=\"stTabsContent\"],"
        "[data-testid=\"stTabs\"] > div:last-child {"
        f"background-color: {bg} !important;"
        f"color: {tx} !important;"
        "border: none !important;"
        "}"

        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar() -> None:
    """Standard sidebar rendered on every page."""
    apply_theme()  # auto day/night based on server clock

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
