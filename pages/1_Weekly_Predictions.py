"""pages/1_📊_Weekly_Predictions.py

Game-by-game predictions for a selected season and week.
Displays model spread, win probability, and O/U vs. book lines.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from utils.ui_components import render_sidebar, themed_dataframe
from utils.storage import FEATURES_DIR, PROCESSED_DIR, load_parquet
from utils.odds_ingestion import build_market_consensus_from_snapshots
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
def _artifact_mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


@st.cache_data(ttl=3600)
def load_feature_matrix(artifact_mtime: int | None = None):
    try:
        return load_parquet("feature_matrix", layer="features")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_market_snapshots(artifact_mtime: int | None = None):
    """Load retained market quotes for display without changing model inputs."""
    try:
        snapshots = load_parquet("line_snapshots")
    except FileNotFoundError:
        return pd.DataFrame()
    if snapshots.empty:
        return pd.DataFrame()
    snapshots["captured_at"] = pd.to_datetime(
        snapshots["captured_at"], utc=True, errors="coerce"
    )
    return snapshots.dropna(subset=["game_id", "captured_at"])


df_all = load_feature_matrix(_artifact_mtime(FEATURES_DIR / "feature_matrix.parquet"))

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
    min_edge = st.slider("Min Edge (spread or O/U pts)", 0.0, 10.0, 0.0, 0.5)
with col3:
    sort_by = st.selectbox("Sort By", ["Edge (High→Low)", "Win Prob", "Game"])

if sel_conf != "All":
    df_week = df_week[
        (df_week["home_conference"] == sel_conf)
        | (df_week["away_conference"] == sel_conf)
    ]

edge_columns = []
if "predicted_spread" in df_week.columns and "market_spread" in df_week.columns:
    df_week["spread_edge"] = (
        df_week["predicted_spread"] + df_week["market_spread"]
    ).abs()
    edge_columns.append("spread_edge")
if "predicted_total" in df_week.columns and "market_total" in df_week.columns:
    df_week["total_edge"] = (
        df_week["predicted_total"] - df_week["market_total"]
    ).abs()
    edge_columns.append("total_edge")

if edge_columns:
    # A game is actionable when either market has an edge. Use the larger
    # available edge so the filter does not discard valid O/U opportunities
    # merely because the spread model is anchored to the market.
    df_week["edge"] = df_week[edge_columns].max(axis=1, skipna=True)
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

# ── retained market quotes ───────────────────────────────────────────────────
market_snapshots = load_market_snapshots(
    _artifact_mtime(PROCESSED_DIR / "line_snapshots.parquet")
)
parlay = (
    market_snapshots[market_snapshots["source"].eq("parlay_api")].copy()
    if not market_snapshots.empty and "source" in market_snapshots.columns
    else pd.DataFrame()
)
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
            "and do not drive the model or recommendations. Cards use the retained "
            "market snapshot union as a display-only fallback when model-facing lines "
            "are unavailable."
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

# Use the latest retained quote union only when the model-facing market fields
# are missing. This keeps ParlayAPI shadow prices visible without allowing them
# to silently replace a production consensus already attached to the feature
# matrix.
display_consensus = build_market_consensus_from_snapshots(
    market_snapshots,
    df_week.assign(season=season),
    season=int(season),
    exclude_sources=(),
)
display_consensus = display_consensus.set_index("game_id") if not display_consensus.empty else pd.DataFrame()


def _browser_timezone() -> tuple[ZoneInfo, str]:
    """Return the browser timezone and a compact regional abbreviation."""
    regional_abbreviations = {
        "America/New_York": "ET",
        "America/Chicago": "CT",
        "America/Denver": "MT",
        "America/Los_Angeles": "PT",
        "America/Anchorage": "AKT",
        "Pacific/Honolulu": "HT",
    }
    try:
        timezone_name = st.context.timezone
    except Exception:
        timezone_name = None
    if not timezone_name:
        return ZoneInfo("UTC"), "UTC"
    try:
        return ZoneInfo(timezone_name), regional_abbreviations.get(
            timezone_name, timezone_name.rsplit("/", 1)[-1].replace("_", " ")
        )
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC"), "UTC"


browser_tz, browser_tz_name = _browser_timezone()
kickoff_column = f"Kickoff ({browser_tz_name})"

compact_rows = []
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
    try:
        fallback = display_consensus.loc[int(row["game_id"])] if not display_consensus.empty else None
    except (KeyError, TypeError, ValueError):
        fallback = None
    if fallback is not None:
        if pd.isna(bs):
            bs = fallback.get("market_spread", float("nan"))
        if pd.isna(bt):
            bt = fallback.get("market_total", float("nan"))
        if pd.isna(hml):
            hml = fallback.get("home_moneyline", float("nan"))
        if pd.isna(aml):
            aml = fallback.get("away_moneyline", float("nan"))

    spread_rec = (
        generate_spread_pick(home, away, ms, bs)
        if pd.notna(ms) and pd.notna(bs) else None
    )
    total_rec = (
        generate_total_pick(home, away, mt, bt)
        if pd.notna(mt) and pd.notna(bt) else None
    )
    ml_rec = (
        generate_moneyline_pick(home, away, float(wp), float(hml), float(aml))
        if pd.notna(hml) and pd.notna(aml) and pd.notna(wp) else None
    )
    kickoff = pd.to_datetime(row.get("start_date"), utc=True, errors="coerce")
    if pd.notna(kickoff):
        kickoff = kickoff.tz_convert(browser_tz)
    kickoff_text = (
        f"{kickoff.strftime('%b')} {kickoff.day} · {kickoff.strftime('%I:%M %p')}"
        if pd.notna(kickoff) else "—"
    )
    hs = row.get("home_score")
    as_ = row.get("away_score")
    result = f"{int(hs)}–{int(as_)}" if pd.notna(hs) and pd.notna(as_) else "—"

    compact_rows.append(
        {
            "Game": f"{away} @ {home}",
            kickoff_column: kickoff_text,
            "Home win": f"{wp:.0%}" if pd.notna(wp) else "—",
            "Model margin": f"{ms:+.1f}" if pd.notna(ms) else "—",
            "Book spread": f"{bs:+.1f}" if pd.notna(bs) else "—",
            "Spread edge": (
                f"{spread_rec.edge:.1f} {CONFIDENCE_EMOJI[spread_rec.confidence]}"
                if spread_rec else "—"
            ),
            "Spread pick": (
                spread_rec.pick if spread_rec and spread_rec.confidence != Confidence.NONE
                else "No edge" if spread_rec else "—"
            ),
            "Model O/U": f"{mt:.1f}" if pd.notna(mt) else "—",
            "Book O/U": f"{bt:.1f}" if pd.notna(bt) else "—",
            "O/U edge": (
                f"{total_rec.edge:.1f} {CONFIDENCE_EMOJI[total_rec.confidence]}"
                if total_rec else "—"
            ),
            "O/U pick": total_rec.pick if total_rec else "—",
            "Home ML": f"{int(hml):+d}" if pd.notna(hml) else "—",
            "Away ML": f"{int(aml):+d}" if pd.notna(aml) else "—",
            "ML edge": f"{ml_rec.edge:.1%}" if ml_rec else "—",
            "ML pick": ml_rec.pick if ml_rec else "—",
            "Result": result,
        }
    )

st.caption(
    f"Compact view — kickoff times shown in {browser_tz_name}. "
    "Use the table’s horizontal scroll to see every market and recommendation."
)
themed_dataframe(
    pd.DataFrame(compact_rows),
    width="stretch",
    height=min(640, max(140, 36 + 35 * len(compact_rows))),
    hide_index=True,
)

add_betting_oracle_footer()
