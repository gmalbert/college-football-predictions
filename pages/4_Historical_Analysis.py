"""pages/4_📈_Historical_Analysis.py

Multi-tab historical analysis:
  • Season Trends — scoring, ATS record by conference, home-field advantage
  • H2H Lookup    — head-to-head history between any two teams
  • Conference Power — cross-conference win rates
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.ui_components import render_sidebar
from utils.storage import load_parquet
from utils.models import predict_batch, models_trained
from footer import add_betting_oracle_footer


render_sidebar()
st.title("📈 Historical Analysis")


# ── load data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    try:
        fm   = load_parquet("feature_matrix", layer="features")
        if models_trained():
            fm = predict_batch(fm)
        return fm
    except FileNotFoundError:
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.warning("No data available. Go to ⚙️ **Settings** and pull historical data first.")
    st.stop()

# ── season filter ─────────────────────────────────────────────────────────────
seasons = sorted(df["season"].dropna().unique())
col1, col2 = st.columns(2)
with col1:
    s_from = st.selectbox("Season From", seasons, index=0)
with col2:
    s_to   = st.selectbox("Season To", seasons, index=len(seasons) - 1)

df_slice = df[(df["season"] >= s_from) & (df["season"] <= s_to)].copy()

tab_trends, tab_h2h, tab_conf = st.tabs(
    ["📅 Season Trends", "🆚 H2H Lookup", "🏆 Conference Power"]
)

# ─────────────────────────────────────────────────────────────────────────────
with tab_trends:
    st.subheader("Season Trends")

    # Avg scoring per season
    scoring = (
        df_slice.groupby("season")
        .agg(avg_total=("total_points", "mean"), avg_margin=("home_margin", "mean"))
        .reset_index()
    )

    fig_score = px.line(
        scoring, x="season", y="avg_total",
        title="Average Total Points per Game by Season",
        markers=True, color_discrete_sequence=["#D4001C"],
    )
    fig_score.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
    )
    st.plotly_chart(fig_score, width="stretch")

    # Home-field advantage over time — avg home margin
    fig_hfa = px.bar(
        scoring, x="season", y="avg_margin",
        title="Average Home Margin by Season (positive = home advantage)",
        color_discrete_sequence=["#D4001C"],
    )
    fig_hfa.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_hfa.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
    )
    st.plotly_chart(fig_hfa, width="stretch")

    # ATS record by conference (if model predictions available)
    if "predicted_spread" in df_slice.columns and "market_spread" in df_slice.columns:
        df_slice["pred_covers"] = df_slice["home_margin"] > -df_slice["market_spread"]
        df_slice["model_picked_home"] = df_slice["predicted_spread"] > df_slice["market_spread"]
        df_slice["ats_correct"] = df_slice["pred_covers"] == df_slice["model_picked_home"]

        ats_by_conf = (
            df_slice.dropna(subset=["home_conference", "ats_correct"])
            .groupby("home_conference")["ats_correct"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "ATS Win %", "count": "Bets", "home_conference": "Conference"})
            .sort_values("ATS Win %", ascending=False)
        )
        st.subheader("Model ATS Win % by Conference")
        st.dataframe(
            ats_by_conf.assign(**{"ATS Win %": lambda d: (d["ATS Win %"] * 100).round(1)}),
            width="stretch", hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
with tab_h2h:
    st.subheader("Head-to-Head Lookup")

    all_teams = sorted(set(df["home_team"].dropna().tolist() + df["away_team"].dropna().tolist()))
    c1, c2 = st.columns(2)
    with c1:
        team_a = st.selectbox("Team A", all_teams, key="h2h_a",
                              index=all_teams.index("Ohio State") if "Ohio State" in all_teams else 0)
    with c2:
        team_b = st.selectbox("Team B", all_teams, key="h2h_b",
                              index=all_teams.index("Michigan") if "Michigan" in all_teams else 1)

    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b))
        | ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    )
    h2h = df[mask].sort_values("season", ascending=False)

    if h2h.empty:
        st.info(f"No matchups found between {team_a} and {team_b} in the dataset.")
    else:
        # All-time record
        a_wins = int(
            ((h2h["home_team"] == team_a) & (h2h["home_margin"] > 0)).sum()
            + ((h2h["away_team"] == team_a) & (h2h["home_margin"] < 0)).sum()
        )
        b_wins = len(h2h) - a_wins

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a} wins", a_wins)
        m2.metric(f"{team_b} wins", b_wins)
        m3.metric("Games", len(h2h))

        rows = []
        for _, g in h2h.iterrows():
            is_a_home = g["home_team"] == team_a
            a_score   = int(g["home_score"] if is_a_home else g["away_score"]) if pd.notna(g.get("home_score")) else None
            b_score   = int(g["away_score"] if is_a_home else g["home_score"]) if pd.notna(g.get("away_score")) else None
            result    = "—"
            if a_score is not None:
                result = f"{team_a} {a_score}–{b_score} {team_b}"
            rows.append({
                "Season": int(g["season"]),
                "Week":   int(g.get("week", 0)),
                "Site":   "Neutral" if g.get("neutral_site") else ("Home" if is_a_home else "Away"),
                "Result": result,
                "Spread": f"{g.get('market_spread', float('nan')):+.1f}" if pd.notna(g.get("market_spread")) else "—",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
with tab_conf:
    st.subheader("Conference Power")

    # Non-conference win rate heatmap per conference per season
    df_nc = df_slice[df_slice["conference_game"] == 0].copy()
    if df_nc.empty:
        st.info("No non-conference game data available in the selected range.")
    else:
        df_nc["home_conf"] = df_nc["home_conference"].fillna("Unknown")
        df_nc["away_conf"] = df_nc["away_conference"].fillna("Unknown")

        # Win rate from home-conference perspective
        pivot_rows = []
        home_conf_groups = df_nc.groupby("home_conf")
        for conf, grp in home_conf_groups:
            wins  = (grp["home_margin"] > 0).sum()
            total = len(grp)
            pivot_rows.append({"Conference": conf, "NC Win %": round(wins / total * 100, 1) if total else 0, "Games": total})

        conf_df = pd.DataFrame(pivot_rows).sort_values("NC Win %", ascending=False)
        fig_bar = px.bar(
            conf_df, x="Conference", y="NC Win %",
            title="Non-Conference Win % by Conference (home games)",
            color="NC Win %", color_continuous_scale="RdYlGn",
            range_color=[30, 70],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
        )
        st.plotly_chart(fig_bar, width="stretch")

        # Avg scoring by conference
        conf_scoring = (
            df_slice.groupby("home_conference")[["home_score", "away_score"]]
            .mean()
            .reset_index()
            .rename(columns={
                "home_conference": "Conference",
                "home_score":      "Avg Home Score",
                "away_score":      "Avg Away Score",
            })
            .dropna()
            .sort_values("Avg Home Score", ascending=False)
        )
        st.subheader("Average Scoring by Conference")
        st.dataframe(
            conf_scoring.round(1),
            width="stretch", hide_index=True,
        )

add_betting_oracle_footer()
