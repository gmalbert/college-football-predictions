"""pages/3_🏟️_Team_Explorer.py

Per-team dashboard: Elo history, advanced-stats radar, and schedule/results.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.ui_components import render_sidebar, themed_dataframe
from utils.storage import load_parquet
from utils.models import predict_batch, models_trained
from footer import add_betting_oracle_footer


render_sidebar()
st.title("🏟️ Team Explorer")


# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all():
    out = {}
    for name, layer in [
        ("feature_matrix", "features"),
        ("elo_ratings",     "processed"),
        ("advanced_stats",  "processed"),
        ("ratings",         "processed"),
    ]:
        try:
            out[name] = load_parquet(name, layer=layer)
        except FileNotFoundError:
            out[name] = pd.DataFrame()
    return out


data = load_all()
fm   = data["feature_matrix"]

if fm.empty:
    st.warning(
        "No data available. Go to ⚙️ **Settings** and pull historical data first."
    )
    st.stop()

# ── selectors ────────────────────────────────────────────────────────────────
all_teams = sorted(
    set(fm["home_team"].dropna().tolist() + fm["away_team"].dropna().tolist())
)
seasons   = sorted(fm["season"].dropna().unique(), reverse=True)

col1, col2 = st.columns([3, 1])
with col1:
    team   = st.selectbox("Team", all_teams, index=all_teams.index("Alabama") if "Alabama" in all_teams else 0)
with col2:
    season = st.selectbox("Season", seasons)

# ── filter games for this team/season ────────────────────────────────────────
mask = (
    ((fm["home_team"] == team) | (fm["away_team"] == team))
    & (fm["season"] == season)
)
df_team = fm[mask].copy().sort_values("week")

# ── team card (summary row) ───────────────────────────────────────────────────
home_games = df_team[df_team["home_team"] == team]
away_games = df_team[df_team["away_team"] == team]

wins   = int(
    (home_games["home_margin"] > 0).sum()
    + (away_games["home_margin"] < 0).sum()
)
losses = len(df_team) - wins

rat = data["ratings"]
rat_row = rat[(rat["team"] == team) & (rat["season"] == season)]
sp_plus = rat_row["sp_overall"].values[0] if not rat_row.empty else None
talent  = rat_row["talent"].values[0]     if not rat_row.empty else None
conf_val = df_team["home_conference"].dropna().mode()
conference = conf_val.iloc[0] if not conf_val.empty else "—"

st.subheader(team)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Record",      f"{wins}‑{losses}")
c2.metric("Conference",  conference)
c3.metric("SP+",         f"{sp_plus:.1f}" if sp_plus else "—")
c4.metric("Talent",      f"{talent:.0f}"  if talent  else "—")
c5.metric("Games",       len(df_team))

st.divider()

# ── Elo history chart ─────────────────────────────────────────────────────────
# elo_ratings has one end-of-season Elo value per (season, team) — no week column.
# Show season-over-season Elo trend for the selected team.
elo_df = data["elo_ratings"]
if not elo_df.empty:
    elo_team = elo_df[elo_df["team"] == team].sort_values("season")
    if not elo_team.empty:
        fig_elo = go.Figure()
        fig_elo.add_trace(go.Scatter(
            x=elo_team["season"], y=elo_team["elo"],
            mode="lines+markers",
            line=dict(color="#D4001C", width=2),
            marker=dict(size=7),
            name="Elo",
        ))
        fig_elo.add_hline(y=1500, line_dash="dot", line_color="gray",
                          annotation_text="Average (1500)")
        fig_elo.update_layout(
            title=f"{team} End-of-Season Elo Rating",
            xaxis_title="Season", yaxis_title="Elo",
            paper_bgcolor="#F7FBFF",
            plot_bgcolor="#F7FBFF",
            font=dict(color="#1A2B3C"),
        )
        st.plotly_chart(fig_elo, width="stretch")
    else:
        st.info("Elo history not available for this team.")
else:
    st.info("Elo data not loaded.")

st.divider()

# ── Advanced stats radar ──────────────────────────────────────────────────────
adv = data["advanced_stats"]
if not adv.empty:
    adv_row = adv[(adv["team"] == team) & (adv["season"] == season)]
    if not adv_row.empty:
        row = adv_row.iloc[0]
        # Build radar of normalized metrics (percentile rank within season)
        radar_metrics = {
            "Off EPA": "off_epa",
            "Off Success": "off_success_rate",
            "Off Explosive": "off_explosiveness",
            "Def EPA (inv)": "def_epa",     # inverted: lower = better
            "Def Success (inv)": "def_success_rate",
        }
        # Percentile ranks within the same season
        adv_season = adv[adv["season"] == season]
        radar_vals = []
        radar_cats = []
        for label, col in radar_metrics.items():
            if col in adv_season.columns and pd.notna(row.get(col)):
                series = adv_season[col].dropna()
                val    = float(row[col])
                pct    = (series < val).mean()    # percentile 0-1
                if "inv" in label.lower():
                    pct = 1 - pct  # lower is better → flip
                radar_vals.append(round(pct * 100, 1))
                radar_cats.append(label)

        if radar_vals:
            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_cats + [radar_cats[0]],
                fill="toself",
                fillcolor="rgba(212,0,28,0.25)",
                line=dict(color="#D4001C"),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100])),
                title=f"{team} Advanced Stats Percentile Rank — {season}",
                paper_bgcolor="#F7FBFF",
                font=dict(color="#1A2B3C"),
            )
            st.plotly_chart(fig_radar, width="stretch")

st.divider()

# ── Schedule & results table ───────────────────────────────────────────────────
st.subheader("Schedule & Results")
if models_trained():
    df_team = predict_batch(df_team)

rows = []
for _, g in df_team.iterrows():
    is_home   = g["home_team"] == team
    opponent  = g["away_team"] if is_home else g["home_team"]
    hs, as_   = g.get("home_score"), g.get("away_score")

    if pd.notna(hs) and pd.notna(as_):
        team_score = int(hs if is_home else as_)
        opp_score  = int(as_ if is_home else hs)
        result     = "W" if team_score > opp_score else "L"
        margin     = team_score - opp_score
    else:
        team_score = opp_score = result = margin = None

    book_spread = g.get("market_spread")
    if pd.notna(book_spread) and pd.notna(margin):
        ats = "✅" if (is_home and margin > -book_spread) or (not is_home and -margin > book_spread) else "❌"
    else:
        ats = "—"

    ms = g.get("predicted_spread")
    rows.append({
        "Wk":       int(g.get("week", 0)),
        "Opponent": opponent,
        "H/A":      "H" if is_home else "A",
        "Result":   f"{result} {team_score}‑{opp_score}" if result else "—",
        "Margin":   f"{margin:+d}" if margin is not None else "—",
        "Book":     f"{book_spread:+.1f}" if pd.notna(book_spread) else "—",
        "ATS":      ats,
        "Model":    f"{ms:+.1f}" if pd.notna(ms) else "—",
    })

if rows:
    themed_dataframe(pd.DataFrame(rows).reset_index(drop=True),
                    width="stretch", hide_index=True)
else:
    st.info("No schedule data for this team and season.")

add_betting_oracle_footer()
