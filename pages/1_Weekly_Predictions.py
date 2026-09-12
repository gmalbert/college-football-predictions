"""pages/1_📊_Weekly_Predictions.py

Game-by-game predictions for a selected season and week.
Displays model spread, win probability, and O/U vs. book lines.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.ui_components import render_sidebar, themed_dataframe
from utils.storage import load_parquet
from utils.models import load_metrics, predict_for_display, models_trained
from utils.betting import (
    generate_spread_pick, generate_total_pick, generate_moneyline_pick,
    CONFIDENCE_EMOJI, Confidence,
)
from footer import add_betting_oracle_footer


render_sidebar()
st.title("📊 Weekly Predictions")
release = load_metrics().get("release_decision", {})
if release.get("decision") != "promote":
    st.warning("Research mode: model release gates have not approved betting use.")

# ── data availability check ──────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_feature_matrix():
    try:
        return load_parquet("feature_matrix", layer="features")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_parlay_snapshots():
    """Load locally captured ParlayAPI quotes without touching model inputs."""
    try:
        snapshots = load_parquet("line_snapshots")
    except FileNotFoundError:
        return pd.DataFrame()
    if snapshots.empty or "source" not in snapshots.columns:
        return pd.DataFrame()
    snapshots = snapshots[snapshots["source"].eq("parlay_api")].copy()
    if snapshots.empty:
        return snapshots
    snapshots["captured_at"] = pd.to_datetime(
        snapshots["captured_at"], utc=True, errors="coerce"
    )
    return snapshots.dropna(subset=["game_id", "captured_at"])


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

season_games = df_all[df_all["season"] == season].copy()
weeks = sorted(season_games["week"].dropna().unique())

# Choose the current calendar week when possible. The old behavior selected
# the highest scheduled week, which jumped to a future Week 15 game in August.
now = pd.Timestamp.now(tz="UTC")
season_games["_start"] = pd.to_datetime(season_games["start_date"], utc=True, errors="coerce")
past_weeks = sorted(season_games.loc[season_games["_start"] <= now, "week"].dropna().unique())
default_week = int(past_weeks[-1]) if past_weeks else int(weeks[0]) if weeks else 1
week = st.selectbox(
    "Week",
    weeks,
    index=weeks.index(default_week) if default_week in weeks else 0,
    format_func=lambda w: f"Week {int(w)}",
)

st.caption(f"Season {int(season)} = the {int(season)} fall schedule · Week {int(week)}")

df_week = df_all[(df_all["season"] == season) & (df_all["week"] == week)].copy()

# ── run predictions ──────────────────────────────────────────────────────────
if models_trained():
    df_week = predict_for_display(df_week)
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
    df_week["edge"] = (df_week["predicted_spread"] + df_week["market_spread"]).abs()
    if min_edge > 0:
        df_week = df_week[df_week["edge"] >= min_edge]
    if sort_by == "Edge (High→Low)":
        df_week = df_week.sort_values("edge", ascending=False)
    elif sort_by == "Win Prob":
        df_week = df_week.sort_values("win_prob", ascending=False)

st.markdown(f"**{len(df_week)} games** — Season {season} · Week {int(week)}")
scopes = set(df_week.get("prediction_scope", pd.Series(dtype=str)).dropna())
if "walk_forward_oos" in scopes:
    st.caption("Completed games use predictions made by season walk-forward backtests; unplayed games use the current full-history model.")
st.divider()

# ── game cards ────────────────────────────────────────────────────────────────
if df_week.empty:
    st.info("No games match the current filters.")
    st.stop()

# ── ParlayAPI shadow quotes ──────────────────────────────────────────────────
parlay = load_parlay_snapshots()
if not parlay.empty:
    week_ids = pd.to_numeric(df_week["game_id"], errors="coerce")
    parlay_week = parlay[
        pd.to_numeric(parlay["game_id"], errors="coerce").isin(week_ids)
    ].copy()
else:
    parlay_week = pd.DataFrame()

if parlay_week.empty:
    st.caption("ParlayAPI shadow quotes: no locally captured prices for this week.")
else:
    game_labels = df_week[["game_id", "away_team", "home_team"]].copy()
    game_labels["game_id"] = pd.to_numeric(game_labels["game_id"], errors="coerce")
    parlay_week["game_id"] = pd.to_numeric(parlay_week["game_id"], errors="coerce")
    parlay_week = parlay_week.merge(game_labels, on="game_id", how="left", validate="many_to_one")
    parlay_week = (
        parlay_week.sort_values("captured_at")
        .drop_duplicates(["game_id", "sportsbook", "market", "side"], keep="last")
    )
    parlay_week["Game"] = (
        parlay_week["away_team"].astype(str)
        + " @ "
        + parlay_week["home_team"].astype(str)
    )
    parlay_week["Market"] = parlay_week["market"].astype(str).str.title()
    parlay_week["Side"] = parlay_week["side"].astype(str).str.title()
    parlay_week["Line"] = pd.to_numeric(parlay_week["line"], errors="coerce").map(
        lambda value: f"{value:+.1f}" if pd.notna(value) else "—"
    )
    parlay_week["American Odds"] = pd.to_numeric(
        parlay_week["odds"], errors="coerce"
    ).map(lambda value: f"{int(value):+d}" if pd.notna(value) else "—")
    parlay_week["Captured UTC"] = parlay_week["captured_at"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    if "provider_observed_at" in parlay_week.columns:
        provider_time = pd.to_datetime(
            parlay_week["provider_observed_at"], utc=True, errors="coerce"
        )
        parlay_week["Provider UTC"] = provider_time.dt.strftime("%Y-%m-%d %H:%M")
        parlay_week["Provider UTC"] = parlay_week["Provider UTC"].fillna("—")
    else:
        parlay_week["Provider UTC"] = "—"
    parlay_week["Live"] = parlay_week["is_live"].fillna(False).map(
        lambda value: "Yes" if bool(value) else "No"
    ) if "is_live" in parlay_week.columns else "No"
    if "stale_seconds" in parlay_week.columns:
        parlay_week["Stale Seconds"] = pd.to_numeric(
            parlay_week["stale_seconds"], errors="coerce"
        ).map(lambda value: f"{value:.0f}" if pd.notna(value) else "—")
    else:
        parlay_week["Stale Seconds"] = "—"

    latest_capture = parlay_week["captured_at"].max().strftime("%Y-%m-%d %H:%M UTC")
    with st.expander(
        f"ParlayAPI shadow quotes · {len(parlay_week):,} latest book/market prices",
        expanded=False,
    ):
        st.caption(
            f"Latest local capture: {latest_capture}. These prices are informational only "
            "and do not drive the model or recommendations."
        )
        themed_dataframe(
            parlay_week[
                [
                    "Game", "sportsbook", "Market", "Side", "Line",
                    "American Odds", "Captured UTC", "Provider UTC", "Live", "Stale Seconds",
                ]
            ].rename(columns={"sportsbook": "Book"})
            .sort_values(["Game", "Book", "Market", "Side"])
            .reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )

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
            m2.metric("Model Home Margin", f"{ms:+.1f}")
        else:
            m2.metric("Model Spread", "—")

        if pd.notna(bs):
            m3.metric("Book Spread", f"{bs:+.1f}")
        else:
            m3.metric("Book Spread", "—")

        # Edge
        if pd.notna(ms) and pd.notna(bs):
            edge_val = ms + bs
            rec      = generate_spread_pick(home, away, ms, bs)
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
