"""pages/2_💰_Value_Bets.py

Surface games where the model line differs materially from the book line.
Includes a bankroll simulator for strategy visualization.
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
    kelly_fraction, half_kelly, simulate_bankroll,
    Confidence, CONFIDENCE_EMOJI, CONFIDENCE_LABEL,
)
from footer import add_betting_oracle_footer


render_sidebar()
st.title("💰 Value Bets")
st.caption("Games where the model's projected line disagrees materially with the sportsbook.")

# ── load data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    try:
        return load_parquet("feature_matrix", layer="features")
    except FileNotFoundError:
        return pd.DataFrame()


df_all = load_data()

if df_all.empty or not models_trained():
    st.warning(
        "No data or models found. Go to ⚙️ **Settings** → "
        "**Pull Historical Data** → **Train Models**."
    )
    st.stop()

# ── controls ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    seasons   = sorted(df_all["season"].dropna().unique(), reverse=True)
    season    = st.selectbox("Season", seasons)
with col2:
    bet_type  = st.selectbox("Bet Type", ["Spread", "Total", "Moneyline", "All"])
with col3:
    min_edge  = st.slider("Min Edge (pts)", 0.5, 10.0, 2.0, 0.5)
with col4:
    min_conf  = st.selectbox(
        "Min Confidence",
        ["All", "LEAN", "MODERATE", "STRONG"],
        index=2,
    )

df_season = df_all[df_all["season"] == season].copy()
df_season = predict_batch(df_season)

# ── helpers ─────────────────────────────────────────────────────────────────
def _conf_gte(c: Confidence, minimum: Confidence) -> bool:
    order = [Confidence.LEAN, Confidence.MODERATE, Confidence.STRONG]
    return order.index(c) >= order.index(minimum)


def _spread_result(margin: float, book: float) -> str:
    if margin > -book:
        return "✅ WIN"
    elif margin == -book:
        return "➡️ PUSH"
    return "❌ LOSS"


def _total_result(total: float, book: float, model: float) -> str:
    over = model > book
    if (over and total > book) or (not over and total < book):
        return "✅ WIN"
    elif total == book:
        return "➡️ PUSH"
    return "❌ LOSS"


def _ml_result(home: str, away: str, margin: float, pick: str) -> str:
    if pd.isna(margin):
        return "—"
    home_won = margin > 0
    away_won = margin < 0
    if home in pick:
        return "✅ WIN" if home_won else ("➡️ PUSH" if margin == 0 else "❌ LOSS")
    if away in pick:
        return "✅ WIN" if away_won else ("➡️ PUSH" if margin == 0 else "❌ LOSS")
    return "—"


# ── build recommendations ────────────────────────────────────────────────────
CONF_ORDER = {
    "STRONG": Confidence.STRONG,
    "MODERATE": Confidence.MODERATE,
    "LEAN": Confidence.LEAN,
    "ALL": None,
}
min_conf_enum = CONF_ORDER.get(min_conf)

records = []
for _, row in df_season.iterrows():
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    gid  = row.get("game_id")
    ms   = row.get("predicted_spread")
    bs   = row.get("market_spread")
    mt   = row.get("predicted_total")
    bt   = row.get("market_total")
    wp   = row.get("win_prob", 0.5)
    actual_margin = row.get("home_margin")
    actual_total  = row.get("total_points")

    if bet_type in ("Spread", "All") and pd.notna(ms) and pd.notna(bs):
        rec = generate_spread_pick(home, away, ms, bs, game_id=gid, win_prob=wp)
        if rec.edge >= min_edge:
            if min_conf_enum is None or _conf_gte(rec.confidence, min_conf_enum):
                records.append({
                    "Week":       int(row.get("week", 0)),
                    "Game":       f"{away} @ {home}",
                    "Bet Type":   "Spread",
                    "Model":      f"{ms:+.1f}",
                    "Book":       f"{bs:+.1f}",
                    "Edge":       round(rec.edge, 1),
                    "Confidence": CONFIDENCE_LABEL[rec.confidence],
                    "Pick":       rec.pick,
                    "Result":     _spread_result(actual_margin, bs) if pd.notna(actual_margin) else "—",
                    "_conf_val":  rec.confidence,
                })

    if bet_type in ("Total", "All") and pd.notna(mt) and pd.notna(bt):
        rec = generate_total_pick(home, away, mt, bt, game_id=gid)
        if rec.edge >= min_edge:
            if min_conf_enum is None or _conf_gte(rec.confidence, min_conf_enum):
                records.append({
                    "Week":       int(row.get("week", 0)),
                    "Game":       f"{away} @ {home}",
                    "Bet Type":   "Total",
                    "Model":      f"{mt:.1f}",
                    "Book":       f"{bt:.1f}",
                    "Edge":       round(rec.edge, 1),
                    "Confidence": CONFIDENCE_LABEL[rec.confidence],
                    "Pick":       rec.pick,
                    "Result":     _total_result(actual_total, bt, mt) if pd.notna(actual_total) else "—",
                    "_conf_val":  rec.confidence,
                })

    if bet_type in ("Moneyline", "All") and pd.notna(wp):
        hml = row.get("home_moneyline")
        aml = row.get("away_moneyline")
        if pd.notna(hml) and pd.notna(aml):
            rec = generate_moneyline_pick(home, away, wp, float(hml), float(aml), game_id=gid)
            if rec is not None and (min_conf_enum is None or _conf_gte(rec.confidence, min_conf_enum)):
                records.append({
                    "Week":       int(row.get("week", 0)),
                    "Game":       f"{away} @ {home}",
                    "Bet Type":   "Moneyline",
                    "Model":      f"{wp:.1%}",
                    "Book":       f"{int(hml):+d} / {int(aml):+d}",
                    "Edge":       round(rec.edge * 100, 1),
                    "Confidence": CONFIDENCE_LABEL[rec.confidence],
                    "Pick":       rec.pick,
                    "Result":     _ml_result(home, away, actual_margin, rec.pick) if pd.notna(actual_margin) else "—",
                    "_conf_val":  rec.confidence,
                })


# ── results table ─────────────────────────────────────────────────────────────
if not records:
    st.info(f"No bets found with edge ≥ {min_edge} pts and confidence ≥ {min_conf}.")
    st.stop()

df_recs = pd.DataFrame(records).sort_values(["Edge"], ascending=False)

# Summary stats
wins   = (df_recs["Result"] == "✅ WIN").sum()
losses = (df_recs["Result"] == "❌ LOSS").sum()
total  = wins + losses

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Bets",   len(df_recs))
m2.metric("Record",       f"{wins}–{losses}" if total else "N/A")
m3.metric("Win Rate",     f"{wins/total:.1%}" if total else "—")
m4.metric("Strong Picks", (df_recs["Confidence"] == "STRONG").sum())

st.divider()
st.subheader("All Value Bets")

display_cols = ["Week", "Game", "Bet Type", "Model", "Book", "Edge", "Confidence", "Pick", "Result"]
st.dataframe(
    df_recs[display_cols].reset_index(drop=True),
    width="stretch",
    hide_index=True,
    column_config={
        "Edge": st.column_config.NumberColumn(format="%.1f pts"),
    },
)

# ── bankroll simulator ────────────────────────────────────────────────────────
st.divider()
st.subheader("💸 Bankroll Simulator")
b1, b2, b3 = st.columns(3)
with b1:
    start_bankroll = st.number_input("Starting Bankroll ($)", value=1000, min_value=100, step=100)
with b2:
    stake_method = st.selectbox("Stake Method", ["Flat (1%)", "Half Kelly", "Full Kelly"])
with b3:
    bet_odds = st.number_input("Odds (American)", value=-110, step=5)

# Build simulated bet history from wins/losses in table
sim_bets = []
for _, row in df_recs[df_recs["Result"].isin(["✅ WIN", "❌ LOSS"])].iterrows():
    wp = 0.53  # assume 53% based on typical model edge
    if stake_method == "Half Kelly":
        stake_frac = half_kelly(wp, bet_odds)
    elif stake_method == "Full Kelly":
        stake_frac = kelly_fraction(wp, bet_odds)
    else:
        stake_frac = 0.01
    sim_bets.append({
        "result": "W" if row["Result"] == "✅ WIN" else "L",
        "odds":   bet_odds,
        "stake":  stake_frac,
    })

if sim_bets:
    bankroll_curve = simulate_bankroll(sim_bets, starting=float(start_bankroll))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=bankroll_curve, mode="lines",
        line=dict(color="#D4001C", width=2),
        fill="tozeroy", fillcolor="rgba(212,0,28,0.1)",
        name="Bankroll",
    ))
    fig.add_hline(y=start_bankroll, line_dash="dash",
                  line_color="gray", annotation_text="Starting bankroll")
    fig.update_layout(
        title="Cumulative Bankroll",
        xaxis_title="Bet #",
        yaxis_title="Bankroll ($)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
    )
    st.plotly_chart(fig, width="stretch")

    final = bankroll_curve[-1]
    roi   = (final - start_bankroll) / start_bankroll
    stz1, stz2 = st.columns(2)
    stz1.metric("Final Bankroll", f"${final:,.0f}", f"{roi:+.1%}")
    stz2.metric("Net P&L", f"${final - start_bankroll:+,.0f}")
else:
    st.caption("No completed bets to simulate — check back once the season has results.")

add_betting_oracle_footer()
