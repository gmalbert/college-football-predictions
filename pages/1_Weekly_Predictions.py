"""pages/1_📊_Weekly_Predictions.py

Game-by-game predictions for a selected season and week.
Displays model spread, win probability, and O/U vs. book lines.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.ui_components import render_sidebar
from utils.storage import load_parquet
from utils.models import predict_batch, models_trained
from utils.betting import (
    generate_spread_pick, generate_total_pick, generate_moneyline_pick,
    CONFIDENCE_EMOJI, Confidence,
)
from footer import add_betting_oracle_footer


render_sidebar()
st.title("📊 Weekly Predictions")

# ── data availability check ──────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_feature_matrix():
    try:
        return load_parquet("feature_matrix", layer="features")
    except FileNotFoundError:
        return pd.DataFrame()


df_all = load_feature_matrix()

if df_all.empty:
    st.warning(
        "No prediction data found. Go to ⚙️ **Settings** and click "
        "**Pull Historical Data** then **Train Models** to get started."
    )
    st.stop()

# ── season / week selectors ──────────────────────────────────────────────────
seasons    = sorted(df_all["season"].dropna().unique(), reverse=True)
default_s  = int(seasons[0]) if seasons else 2025
season     = st.selectbox("Season", seasons, index=0)

weeks      = sorted(df_all[df_all["season"] == season]["week"].dropna().unique())
default_w  = int(weeks[-1]) if weeks else 1
week       = st.selectbox("Week", weeks, index=len(weeks) - 1, format_func=lambda w: f"Week {int(w)}")

df_week = df_all[(df_all["season"] == season) & (df_all["week"] == week)].copy()

# ── run predictions ──────────────────────────────────────────────────────────
if models_trained():
    df_week = predict_batch(df_week)
else:
    st.info("Models not yet trained. Go to ⚙️ Settings → Train Models.")
    for col in ["win_prob", "predicted_spread", "predicted_total"]:
        df_week[col] = float("nan")

# ── filters ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    confs = ["All"] + sorted(df_week["home_conference"].dropna().unique().tolist())
    sel_conf = st.selectbox("Conference", confs)
with col2:
    min_edge = st.slider("Min Edge (pts)", 0.0, 10.0, 0.0, 0.5)
with col3:
    sort_by = st.selectbox("Sort By", ["Edge (High→Low)", "Win Prob", "Game"])

if sel_conf != "All":
    df_week = df_week[
        (df_week["home_conference"] == sel_conf)
        | (df_week["away_conference"] == sel_conf)
    ]

if "predicted_spread" in df_week.columns and "market_spread" in df_week.columns:
    df_week["edge"] = (df_week["predicted_spread"] - df_week["market_spread"]).abs()
    if min_edge > 0:
        df_week = df_week[df_week["edge"] >= min_edge]
    if sort_by == "Edge (High→Low)":
        df_week = df_week.sort_values("edge", ascending=False)
    elif sort_by == "Win Prob":
        df_week = df_week.sort_values("win_prob", ascending=False)

st.markdown(f"**{len(df_week)} games** — Season {season} · Week {int(week)}")
st.divider()

# ── game cards ────────────────────────────────────────────────────────────────
if df_week.empty:
    st.info("No games match the current filters.")
    st.stop()

for _, row in df_week.iterrows():
    home = row.get("home_team", "—")
    away = row.get("away_team", "—")
    wp   = row.get("win_prob", float("nan"))
    ms   = row.get("predicted_spread", float("nan"))
    bs   = row.get("market_spread", float("nan"))
    mt   = row.get("predicted_total", float("nan"))
    bt   = row.get("market_total", float("nan"))
    hml  = row.get("home_moneyline", float("nan"))
    aml  = row.get("away_moneyline", float("nan"))

    with st.container():
        hdr1, hdr2, hdr3 = st.columns([5, 1, 5])
        with hdr1:
            st.subheader(away)
            st.caption("Away")
        with hdr2:
            st.markdown("### @")
        with hdr3:
            st.subheader(home)
            st.caption("Home")

        m1, m2, m3, m4, m5 = st.columns(5)

        # Win probability
        if pd.notna(wp):
            wp_color = "🟢" if wp >= 0.65 else "🟡" if wp >= 0.50 else "🔴"
            m1.metric("Home Win Prob", f"{wp:.0%} {wp_color}")
        else:
            m1.metric("Home Win Prob", "—")

        # Model spread vs book spread
        if pd.notna(ms):
            m2.metric("Model Spread", f"{ms:+.1f}")
        else:
            m2.metric("Model Spread", "—")

        if pd.notna(bs):
            m3.metric("Book Spread", f"{bs:+.1f}")
        else:
            m3.metric("Book Spread", "—")

        # Edge
        if pd.notna(ms) and pd.notna(bs):
            edge_val = ms - bs
            rec      = generate_spread_pick(home, away, ms, bs, win_prob=wp if pd.notna(wp) else 0.5)
            badge    = CONFIDENCE_EMOJI[rec.confidence]
            m4.metric("Spread Edge", f"{abs(edge_val):.1f} {badge}")
            m5.metric("Pick", rec.pick if rec.confidence.value != "none" else "No edge")
        else:
            m4.metric("Edge", "—")
            m5.metric("Pick", "—")

        # O/U row
        if pd.notna(mt) or pd.notna(bt):
            t1, t2, t3, t4 = st.columns([2, 2, 3, 3])
            t1.metric("Model O/U", f"{mt:.1f}" if pd.notna(mt) else "—")
            t2.metric("Book O/U",  f"{bt:.1f}" if pd.notna(bt) else "—")
            if pd.notna(mt) and pd.notna(bt):
                total_rec = generate_total_pick(home, away, mt, bt)
                badge = CONFIDENCE_EMOJI[total_rec.confidence]
                t3.metric("O/U Pick", total_rec.pick)
                t4.metric("O/U Edge", f"{total_rec.edge:.1f} pts {badge}")

        # Moneyline row
        if pd.notna(hml) and pd.notna(aml) and pd.notna(wp):
            ml_rec = generate_moneyline_pick(home, away, wp, float(hml), float(aml))
            if ml_rec and ml_rec.confidence != Confidence.NONE:
                u1, u2, u3, u4 = st.columns([2, 2, 3, 3])
                u1.metric("Home ML", f"{int(hml):+d}")
                u2.metric("Away ML", f"{int(aml):+d}")
                badge = CONFIDENCE_EMOJI[ml_rec.confidence]
                u3.metric("ML Pick", ml_rec.pick)
                u4.metric("ML Edge", f"{ml_rec.edge:.1%} {badge}")

        # Actual result (historical data)
        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            st.caption(
                f"Result: {home} {int(hs)} – {int(as_)} {away}  "
                f"(margin {int(hs) - int(as_):+d})"
            )

        st.divider()

add_betting_oracle_footer()
