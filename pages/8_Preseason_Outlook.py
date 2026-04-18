"""pages/8_Preseason_Outlook.py

Preseason / returning production dashboard:
  • Returning production leaderboard (% of PPA returning)
  • Transfer portal net impact by team
  • Team efficiency scatter: off EPA vs def EPA
"""
from __future__ import annotations

import streamlit as st
import plotly.express as px
import pandas as pd

from utils.storage import load_parquet
from utils.logger import get_logger
from utils.ui_components import render_sidebar, themed_dataframe

logger = get_logger(__name__)

render_sidebar()

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    try:
        ret_df = load_parquet("returning_production")
        avail_seasons = sorted(ret_df["season"].dropna().unique().tolist(), reverse=True)
    except FileNotFoundError:
        avail_seasons = list(range(2025, 2020, -1))
        ret_df = pd.DataFrame()

    season = st.selectbox("Season", avail_seasons)

    try:
        adv_df = load_parquet("advanced_stats")
    except FileNotFoundError:
        adv_df = pd.DataFrame()

    try:
        portal_df = load_parquet("transfer_portal")
    except FileNotFoundError:
        portal_df = pd.DataFrame()

st.title("🏈 Preseason Outlook")
st.caption(
    "Returning production and transfer portal data — "
    "critical context for early-season predictions."
)

# ─────────────────────────────── Tabs ────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Returning Production",
    "🔄 Transfer Portal",
    "⚡ Team Efficiency Quadrant",
])

# ─────────────────────────────── Tab 1: Returning Production ─────────────────
with tab1:
    st.subheader(f"{season} — Returning Production Leaderboard")
    if ret_df.empty:
        st.info("No returning production data. Run the data pipeline first.")
    else:
        df = ret_df[ret_df["season"] == season].copy()
        if df.empty:
            st.warning(f"No returning production data for {season}.")
        else:
            # Pct columns may be 0-1 or 0-100; normalise to 0-100
            for col in ["percent_ppa", "percent_passing_ppa",
                        "percent_receiving_ppa", "percent_rushing_ppa"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if df[col].median() < 2:          # likely 0–1 scale
                        df[col] = df[col] * 100

            df = df.sort_values("percent_ppa", ascending=False)

            col_cfg: dict = {}
            if "percent_ppa" in df.columns:
                col_cfg["Total PPA Returning %"] = st.column_config.ProgressColumn(
                    "Total PPA Returning %", min_value=0, max_value=100, format="%.1f%%"
                )
            if "percent_passing_ppa" in df.columns:
                col_cfg["Passing PPA Returning %"] = st.column_config.ProgressColumn(
                    "Passing PPA Returning %", min_value=0, max_value=100, format="%.1f%%"
                )
            if "percent_rushing_ppa" in df.columns:
                col_cfg["Rushing PPA Returning %"] = st.column_config.ProgressColumn(
                    "Rushing PPA Returning %", min_value=0, max_value=100, format="%.1f%%"
                )
            if "percent_receiving_ppa" in df.columns:
                col_cfg["Receiving PPA Returning %"] = st.column_config.ProgressColumn(
                    "Receiving PPA Returning %", min_value=0, max_value=100, format="%.1f%%"
                )

            show_cols = [c for c in
                         ["team", "conference", "percent_ppa", "percent_passing_ppa",
                          "percent_rushing_ppa", "percent_receiving_ppa"]
                         if c in df.columns]
            col_headers = {
                "team": "Team",
                "conference": "Conference",
                "percent_ppa": "Total PPA Returning %",
                "percent_passing_ppa": "Passing PPA Returning %",
                "percent_rushing_ppa": "Rushing PPA Returning %",
                "percent_receiving_ppa": "Receiving PPA Returning %",
            }

            display_df = df[show_cols].head(130).rename(columns=col_headers)
            themed_dataframe(
                display_df,
                column_config=col_cfg,
                width="stretch",
                height=480,
            )

            if "percent_ppa" in df.columns and "team" in df.columns:
                top30 = df[show_cols].head(30).copy()
                fig = px.bar(
                    top30, x="team", y="percent_ppa",
                    color="conference" if "conference" in top30.columns else None,
                    title=f"{season} — Top 30 Teams by Returning PPA %",
                    labels={"percent_ppa": "% of PPA Returning", "team": "Team"},
                )
                fig.update_layout(
                    xaxis_tickangle=-45, height=420,
                    paper_bgcolor="#F7FBFF", plot_bgcolor="#F7FBFF",
                    font=dict(color="#1A2B3C"),
                )
                st.plotly_chart(fig, width="stretch")

# ─────────────────────────────── Tab 2: Transfer Portal ──────────────────────
with tab2:
    st.subheader(f"{season} — Transfer Portal Net Impact")
    if portal_df.empty:
        st.info("No transfer portal data. Run the data pipeline first.")
    else:
        pf = portal_df[portal_df["season"] == season].copy()
        if pf.empty:
            st.warning(f"No portal data for {season}.")
        else:
            pf["rating"] = pd.to_numeric(pf["rating"], errors="coerce").fillna(0)

            gains = (pf.groupby("destination")["rating"]
                     .agg(gains_sum="sum", gains_count="count")
                     .reset_index().rename(columns={"destination": "team"}))
            losses = (pf.groupby("origin")["rating"]
                      .agg(losses_sum="sum", losses_count="count")
                      .reset_index().rename(columns={"origin": "team"}))
            merged = gains.merge(losses, on="team", how="outer").fillna(0)
            merged["net_rating"] = merged["gains_sum"] - merged["losses_sum"]
            merged["net_count"]  = merged["gains_count"] - merged["losses_count"]
            merged = merged.sort_values("net_rating", ascending=False)

            col_a, col_b = st.columns(2)
            gain_cols = {
                "team": "Team",
                "gains_sum": "Gains Rating",
                "gains_count": "Gain Count",
                "losses_sum": "Losses Rating",
                "net_rating": "Net Rating",
            }
            portal_col_cfg = {
                "Gains Rating": st.column_config.NumberColumn(format="%.2f"),
                "Gain Count": st.column_config.NumberColumn(format="%.2f"),
                "Losses Rating": st.column_config.NumberColumn(format="%.2f"),
                "Net Rating": st.column_config.NumberColumn(format="%.2f"),
            }
            with col_a:
                st.markdown("**Top Gainers (by recruit rating sum)**")
                themed_dataframe(
                    merged[["team", "gains_sum", "gains_count",
                             "losses_sum", "net_rating"]]
                    .head(20)
                    .rename(columns=gain_cols),
                    column_config=portal_col_cfg,
                    width="stretch",
                )
            with col_b:
                st.markdown("**Biggest Losers**")
                themed_dataframe(
                    merged[["team", "gains_sum", "gains_count",
                             "losses_sum", "net_rating"]]
                    .tail(20)
                    .rename(columns=gain_cols),
                    column_config=portal_col_cfg,
                    width="stretch",
                )

            fig = px.bar(
                merged.head(30), x="team", y="net_rating",
                color="net_rating",
                color_continuous_scale="RdYlGn",
                title=f"{season} — Top 30 Teams by Transfer Portal Net Rating",
                labels={"net_rating": "Net Rating (Gains − Losses)", "team": "Team"},
            )
            fig.update_layout(
                xaxis_tickangle=-45, height=420, coloraxis_showscale=False,
                paper_bgcolor="#F7FBFF", plot_bgcolor="#F7FBFF",
                font=dict(color="#1A2B3C"),
            )
            st.plotly_chart(fig, width="stretch")

# ─────────────────────────────── Tab 3: EPA Scatter ──────────────────────────
with tab3:
    st.subheader(f"{season} — Team Efficiency Quadrant")
    st.caption("Top-right = elite offense AND elite defense. Source: CFBD advanced stats.")

    if adv_df.empty:
        st.info("No advanced stats data. Run the data pipeline first.")
    else:
        df3 = adv_df[adv_df["season"] == season].copy()
        if df3.empty:
            st.warning(f"No advanced stats for {season}.")
        else:
            for col in ["off_epa", "def_epa"]:
                if col in df3.columns:
                    df3[col] = pd.to_numeric(df3[col], errors="coerce")

            if "off_epa" in df3.columns and "def_epa" in df3.columns:
                if "conference" in df3.columns:
                    conferences = sorted(df3["conference"].dropna().unique().tolist())
                    selected_conf = st.selectbox(
                        "Conference",
                        ["All"] + conferences,
                        index=0,
                        key="efficiency_quadrant_conf",
                    )
                    if selected_conf != "All":
                        df3 = df3[df3["conference"] == selected_conf]
                df3 = df3.dropna(subset=["off_epa", "def_epa"])
                if df3.empty:
                    st.warning(f"No data for {selected_conf} conference in {season}.")
                else:
                    fig3 = px.scatter(
                        df3,
                        x="off_epa",
                        y="def_epa",
                        text="team",
                        color="conference" if "conference" in df3.columns else None,
                        title=f"{season} — Offense EPA/Play vs Defense EPA/Play",
                        labels={
                            "off_epa": "Offensive EPA/Play (higher = better)",
                            "def_epa": "Defensive EPA/Play (lower = better)",
                        },
                        height=580,
                    )
                fig3.update_traces(textposition="top center", marker_size=9)
                fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
                fig3.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
                # Quadrant labels
                x_max = float(df3["off_epa"].max()) * 0.8
                y_min = float(df3["def_epa"].min()) * 0.8
                x_min = float(df3["off_epa"].min()) * 0.8
                y_max = float(df3["def_epa"].max()) * 0.8
                for txt, xx, yy in [
                    ("ELITE", x_max, y_min),
                    ("OFFENSE ONLY", x_max, y_max),
                    ("DEFENSE ONLY", x_min, y_min),
                    ("REBUILDING", x_min, y_max),
                ]:
                    fig3.add_annotation(
                        x=xx, y=yy, text=txt, showarrow=False,
                        font=dict(size=13, color="rgba(100,100,100,0.6)"),
                    )
                fig3.update_layout(
                    margin=dict(l=40, r=40, t=60, b=40),
                    paper_bgcolor="#F7FBFF", plot_bgcolor="#F7FBFF",
                    font=dict(color="#1A2B3C"),
                )
                st.plotly_chart(fig3, width="stretch")
            else:
                st.info("off_epa / def_epa columns not found in advanced stats.")
