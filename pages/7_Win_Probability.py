"""pages/7_Win_Probability.py

Interactive in-game win probability chart for any historical CFBD game.
Fetches play-by-play WP data from the CFBD metrics API.
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.cfbd_client import get_win_probability_chart, get_games
from utils.config import get_secret
from utils.ui_components import render_sidebar, themed_dataframe

render_sidebar()

st.title("📈 In-Game Win Probability")
st.caption(
    "Play-by-play home-team win probability from the CFBD model. "
    "Select a season and week, then pick a game."
)

# ── game selection controls (main page) ──────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 3])
with col1:
    season = st.selectbox("Season", list(range(2025, 2020, -1)), index=0)
with col2:
    season_type_display = st.selectbox("Season type", ["Regular", "Postseason"])
    season_type = season_type_display.lower()
with col3:
    week_options = list(range(1, 18)) if season_type == "regular" else list(range(1, 6))
    week = st.selectbox("Week", week_options)

# ── load games for the chosen week ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def _load_games(year: int, wk: int, stype: str) -> list:
    return get_games(year, season_type=stype, week=wk)


with st.spinner("Loading games…"):
    games = _load_games(season, week, season_type)

if not games:
    st.info("No games found for the selected week.")
    st.stop()

game_options: dict[str, int] = {}
for g in games:
    home = getattr(g, "home_team", None) or g.get("home_team", "?") if isinstance(g, dict) else getattr(g, "home_team", "?")
    away = getattr(g, "away_team", None) or g.get("away_team", "?") if isinstance(g, dict) else getattr(g, "away_team", "?")
    gid  = getattr(g, "id", None)       or g.get("id", None)        if isinstance(g, dict) else getattr(g, "id", None)
    if gid:
        game_options[f"{away} @ {home}"] = gid

if not game_options:
    st.info("No games with IDs found.")
    st.stop()

search = st.text_input("Search teams", placeholder="e.g. Alabama, Ohio State…")
filtered = sorted(
    k for k in game_options if not search or search.lower() in k.lower()
)

if not filtered:
    st.warning(f"No games matching '{search}'.")
    st.stop()

selected_label = st.selectbox("Select game", filtered)
game_id = game_options[selected_label]

st.caption(f"Game ID: `{game_id}`")

# ── fetch WP data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def _load_wp(gid: int) -> list:
    return get_win_probability_chart(gid)


with st.spinner("Fetching win probability data…"):
    wp_data = _load_wp(game_id)

if not wp_data:
    st.warning(
        "No win probability data available for this game. "
        "CFBD only provides this for completed games."
    )
    st.stop()

# ── parse WP data ─────────────────────────────────────────────────────────────
rows: list[dict] = []
for i, p in enumerate(wp_data):
    if isinstance(p, dict):
        home_wp = p.get("homeWinProb") or p.get("home_win_prob")
        play_num = p.get("playNumber") or p.get("play_number") or i
        home_score = p.get("homeScore") or p.get("home_score")
        away_score = p.get("awayScore") or p.get("away_score")
        play_text  = p.get("playText")  or p.get("play_text", "")
    else:
        home_wp    = getattr(p, "home_win_prob", None) or getattr(p, "homeWinProb", None)
        play_num   = getattr(p, "play_number", i) or i
        home_score = getattr(p, "home_score", None)
        away_score = getattr(p, "away_score", None)
        play_text  = getattr(p, "play_text", "")
    if home_wp is not None:
        rows.append({
            "play": int(play_num),
            "home_wp": float(home_wp),
            "home_score": home_score,
            "away_score": away_score,
            "play_text": str(play_text),
        })

if not rows:
    st.warning("Win probability data could not be parsed.")
    st.stop()

df = pd.DataFrame(rows).sort_values("play")

# ── derive team names from selection ──────────────────────────────────────────
parts = selected_label.split(" @ ")
away_name = parts[0] if len(parts) == 2 else "Away"
home_name = parts[1] if len(parts) == 2 else "Home"

# ── chart ─────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
final_wp = float(df["home_wp"].iloc[-1])
col1.metric(f"{home_name} Win Prob", f"{final_wp:.1%}")
col2.metric(f"{away_name} Win Prob", f"{1 - final_wp:.1%}")
if df["home_score"].notna().any():
    col3.metric(
        "Final Score",
        f"{home_name} {int(df['home_score'].iloc[-1] or 0)}  –  "
        f"{int(df['away_score'].iloc[-1] or 0)}  {away_name}",
    )

st.divider()

fig = go.Figure()

# Fill for home team above 50%
fig.add_trace(go.Scatter(
    x=df["play"], y=df["home_wp"],
    fill="tozeroy", fillcolor="rgba(30, 100, 200, 0.15)",
    line=dict(color="rgba(30, 100, 200, 0.8)", width=2),
    name=f"{home_name} WP",
    hovertemplate=(
        "Play %{x}<br>"
        f"{home_name} Win Prob: %{{y:.1%}}<br>"
        "%{customdata}<extra></extra>"
    ),
    customdata=df["play_text"],
))

# 50% line
fig.add_hline(
    y=0.5, line_dash="dash", line_color="gray",
    annotation_text="50%", annotation_position="right",
)

# Quarter markers (approx — every ~18 plays in a typical game)
total_plays = len(df)
for q, label in [(total_plays // 4, "Q2"), (total_plays // 2, "Q3"),
                  (3 * total_plays // 4, "Q4")]:
    fig.add_vline(x=df["play"].iloc[min(q, len(df)-1)],
                  line_dash="dot", line_color="lightgray", opacity=0.6)
    fig.add_annotation(
        x=df["play"].iloc[min(q, len(df)-1)],
        y=0.95, text=label, showarrow=False,
        font=dict(size=10, color="gray"),
    )

fig.update_layout(
    title=f"{away_name} @ {home_name} — Win Probability",
    xaxis_title="Play Number",
    yaxis_title=f"{home_name} Win Probability",
    yaxis=dict(range=[0, 1], tickformat=".0%"),
    height=480,
    showlegend=False,
    hovermode="x unified",
    margin=dict(l=40, r=40, t=50, b=40),
    paper_bgcolor="#F7FBFF",
    plot_bgcolor="#F7FBFF",
)
st.plotly_chart(fig, width="stretch")

# ── raw data expander ────────────────────────────────────────────────────────
with st.expander("Raw play data"):
    themed_dataframe(df, width="stretch", height=300)
