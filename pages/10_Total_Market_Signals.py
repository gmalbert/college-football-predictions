"""Validated retrospective total-side edge and prospective shadow signals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from footer import add_betting_oracle_footer
from utils.models import load_metrics
from utils.storage import DATA_DIR, load_parquet
from utils.ui_components import render_sidebar, themed_dataframe


render_sidebar()
st.title("Total Market Signals")
st.caption(
    "A separate closing-time over/under classifier using opening-to-current movement, "
    "cross-book dispersion, market depth, and point-in-time team form."
)

metrics = load_metrics().get("total_cover_model", {})
strategy = metrics.get("strategy", {})
release = metrics.get("strategy_release", {})
eligible_sides = release.get("eligible_sides", [])

if not metrics:
    st.warning("The total-side artifact has not been trained.")
    st.stop()

status = str(release.get("status", "hold")).upper()
if status == "SHADOW":
    st.warning(
        "Shadow deployment only. Retrospective gates passed, but no signal becomes "
        "an automated bet until 2026 prospective closing-line value passes."
    )
else:
    st.error("The total-side strategy is on hold.")

over_metrics = strategy.get("by_side", {}).get("over", {})
confirmation = (
    strategy.get("by_season_and_side", {}).get("2025", {}).get("over", {})
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("OOS Brier", f"{metrics.get('brier', float('nan')):.4f}", "vs 0.2500")
c2.metric(
    "Validated Over Record",
    f"{over_metrics.get('wins', 0)}–{over_metrics.get('losses', 0)}",
    f"{over_metrics.get('win_rate', 0):.1%}",
)
c3.metric(
    "2025 Confirmation",
    f"{confirmation.get('wins', 0)}–{confirmation.get('losses', 0)}",
    f"{confirmation.get('win_rate', 0):.1%}",
)
c4.metric(
    "Shadow-Eligible Side",
    ", ".join(side.title() for side in eligible_sides) or "None",
)
st.caption(
    f"Selection requires probability ≥ {strategy.get('minimum_selected_probability', 0.575):.1%}. "
    "Historical ROI assumes every price was -110; actual deployment requires captured executable prices."
)

st.subheader("Walk-forward decisions")
try:
    backtest = load_parquet("model_backtest", layer="features")
except FileNotFoundError:
    backtest = pd.DataFrame()

if backtest.empty or "total_over_prob_oos" not in backtest:
    st.info("No total-side OOS artifact is available.")
else:
    threshold = float(strategy.get("probability_edge_threshold", 0.075))
    selected = backtest[
        backtest["total_over_prob_oos"].notna()
        & ((backtest["total_over_prob_oos"] - 0.5).abs() >= threshold)
    ].copy()
    selected["Side"] = np.where(
        selected["total_over_prob_oos"] > 0.5, "Over", "Under"
    )
    selected = selected[selected["Side"].str.lower().isin(eligible_sides)]
    seasons = sorted(selected["season"].dropna().unique(), reverse=True)
    season = st.selectbox("Held-out season", seasons) if seasons else None
    if season is not None:
        selected = selected[selected["season"] == season].copy()
    selected["Game"] = (
        selected.get("away_team", selected["game_id"].astype(str)).astype(str)
        + " @ "
        + selected.get("home_team", "").astype(str)
    )
    selected["Probability"] = np.where(
        selected["Side"].eq("Over"),
        selected["total_over_prob_oos"],
        1 - selected["total_over_prob_oos"],
    )
    selected["Result"] = np.where(
        np.where(
            selected["Side"].eq("Over"),
            selected["total_points"] > selected["market_total"],
            selected["total_points"] < selected["market_total"],
        ),
        "WIN",
        "LOSS",
    )
    columns = [
        column for column in (
            "week", "Game", "Side", "market_total", "market_total_open",
            "market_total_move", "Probability", "Result",
        ) if column in selected
    ]
    themed_dataframe(selected[columns].sort_values("Probability", ascending=False), hide_index=True)

st.subheader("Current prospective shadow file")
shadow_path = DATA_DIR / "shadow_total_signals.json"
shadow = json.loads(shadow_path.read_text(encoding="utf-8")) if shadow_path.exists() else {}
signals = pd.DataFrame(shadow.get("signals", []))
if signals.empty:
    st.info(shadow.get("meta", {}).get("note", "No closing-time shadow signals currently qualify."))
else:
    themed_dataframe(signals, hide_index=True)

add_betting_oracle_footer()
