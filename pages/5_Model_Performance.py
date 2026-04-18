"""pages/5_🎯_Model_Performance.py

Model dashboard: Brier score, ATS record, calibration curve,
feature importance, and rolling weekly performance.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.ui_components import render_sidebar
from utils.storage import load_parquet
from utils.models import load_metrics, models_trained, load_models, WIN_MODEL_PATH
from utils.feature_engine import WIN_FEATURES, SPREAD_FEATURES, TOTAL_FEATURES
from footer import add_betting_oracle_footer

try:
    from sklearn.calibration import calibration_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


render_sidebar()
st.title("🎯 Model Performance")

if not models_trained():
    st.warning(
        "Models not yet trained. Go to ⚙️ **Settings** → **Train Models** first."
    )
    st.stop()

metrics = load_metrics()

# ── top-line metrics ──────────────────────────────────────────────────────────
win_m    = metrics.get("win_model", {})
spread_m = metrics.get("spread_model", {})
total_m  = metrics.get("total_model", {})
ats_m    = metrics.get("ats", {})

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Brier Score",    f"{win_m.get('brier', 0):.4f}",     help="< 0.20 is good")
m2.metric("Spread RMSE",    f"{spread_m.get('rmse', 0):.2f} pts", help="< 14 pts target")
m3.metric("Total RMSE",     f"{total_m.get('rmse', 0):.2f} pts",  help="< 12 pts target")
m4.metric("ATS Win %",      f"{ats_m.get('pct', 0):.1%}",          help="Break-even is 52.4%")
m5.metric("ATS Record",     f"{ats_m.get('wins',0)}‑{ats_m.get('losses',0)}")

st.divider()

# ── load feature matrix for charts ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_fm():
    try:
        return load_parquet("feature_matrix", layer="features")
    except FileNotFoundError:
        return pd.DataFrame()


df = load_fm()

# ─────────────────── calibration curve ───────────────────────────────────────
if not df.empty and HAS_SKLEARN:
    st.subheader("Calibration Curve — Win Probability")
    models = load_models()
    win_mod = models.get("win")

    feats   = [f for f in WIN_FEATURES if f in df.columns]
    df_cal  = df.dropna(subset=feats + ["home_win"]).copy()

    if win_mod is not None and not df_cal.empty:
        X = df_cal[feats].fillna(0).values
        if HAS_XGB and isinstance(win_mod, xgb.Booster):
            probs = win_mod.predict(xgb.DMatrix(X))
        else:
            probs = win_mod.predict_proba(X)[:, 1]

        y_true = df_cal["home_win"].values.astype(int)
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos,
            mode="lines+markers",
            line=dict(color="#D4001C", width=2),
            marker=dict(size=8),
            name="Model",
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Perfect calibration",
        ))
        fig_cal.update_layout(
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            paper_bgcolor="#F7FBFF",
            plot_bgcolor="#F7FBFF",
            font=dict(color="#1A2B3C"),
        )
        st.plotly_chart(fig_cal, width="stretch")
    else:
        st.info("Win model not available for calibration chart.")

    st.divider()

# ─────────────────── ATS record by week ──────────────────────────────────────
if not df.empty and "predicted_spread" in df.columns and "market_spread" in df.columns:
    st.subheader("ATS Record by Week")

    sub = df.dropna(subset=["predicted_spread", "market_spread", "home_margin"]).copy()
    sub["model_picks_home"] = sub["predicted_spread"] > sub["market_spread"]
    sub["home_covered"]     = sub["home_margin"] > -sub["market_spread"]
    sub["correct"]          = sub["model_picks_home"] == sub["home_covered"]

    ats_week = (
        sub.groupby("week")["correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "win_pct", "count": "n", "week": "Week"})
    )

    fig_ats = go.Figure()
    fig_ats.add_trace(go.Bar(
        x=ats_week["Week"], y=ats_week["win_pct"] * 100,
        marker_color=[
            "#D4001C" if v >= 52.4 else "#333" for v in ats_week["win_pct"] * 100
        ],
        name="ATS Win %",
    ))
    fig_ats.add_hline(y=52.4, line_dash="dash", line_color="gold",
                      annotation_text="Break-even (52.4%)")
    fig_ats.update_layout(
        yaxis_title="ATS Win %",
        xaxis_title="Week",
        paper_bgcolor="#F7FBFF",
        plot_bgcolor="#F7FBFF",
        font=dict(color="#1A2B3C"),
    )
    st.plotly_chart(fig_ats, width="stretch")
    st.divider()

# ─────────────────── feature importance ──────────────────────────────────────
st.subheader("Feature Importance — Spread Model")
models = load_models()
spread_mod = models.get("spread")

if spread_mod is not None:
    feat_names = [f for f in SPREAD_FEATURES if True]  # use all defined cols

    if HAS_XGB and isinstance(spread_mod, xgb.Booster):
        raw_imp = spread_mod.get_score(importance_type="gain")
        imp_df  = (
            pd.DataFrame(
                [{"Feature": f"f{i}", "Importance": raw_imp.get(f"f{i}", 0)}
                 for i in range(len(SPREAD_FEATURES))]
            )
            .assign(Feature=lambda d: [SPREAD_FEATURES[int(r["Feature"][1:])]
                                        if int(r["Feature"][1:]) < len(SPREAD_FEATURES)
                                        else r["Feature"]
                                        for _, r in d.iterrows()])
            .sort_values("Importance", ascending=True)
        )
    else:
        # sklearn Ridge / Pipeline — use absolute coefficients
        try:
            coefs = abs(spread_mod.named_steps["ridge"].coef_)
            used_feats = [f for f in SPREAD_FEATURES if f in (df.columns if not df.empty else SPREAD_FEATURES)][:len(coefs)]
            imp_df = pd.DataFrame({"Feature": used_feats, "Importance": coefs}).sort_values("Importance")
        except Exception:
            imp_df = pd.DataFrame()

    if not imp_df.empty:
        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"],
            orientation="h",
            marker_color="#D4001C",
        ))
        fig_imp.update_layout(
            xaxis_title="Importance",
            paper_bgcolor="#F7FBFF",
            plot_bgcolor="#F7FBFF",
            font=dict(color="#1A2B3C"),
        )
        st.plotly_chart(fig_imp, width="stretch")
        with st.expander("Spread Model Feature Dictionary", expanded=False):
            st.markdown(
                """
                | Feature | Description | Why helpful |
                |---|---|---|
                | elo_diff | Home team Elo rating minus away team Elo rating. | Captures overall team strength and matchup quality. |
                | sp_plus_diff | SP+ overall rating difference (home minus away). | Adds a modern analytics rating that blends offense, defense, and schedule. |
                | sp_offense_diff | SP+ offensive efficiency difference. | Highlights which team has the more productive offense. |
                | sp_defense_diff | SP+ defensive efficiency difference. | Reflects defensive capability to limit opponents. |
                | off_epa_diff | Offensive EPA per play difference. | Measures efficiency by scoring value per play. |
                | def_epa_diff | Defensive EPA per play difference. | Captures how well a defense prevents high-value plays. |
                | off_explosiveness_diff | Explosiveness metric difference. | Shows the team likely to generate big plays. |
                | def_havoc_diff | Defensive havoc metric difference. | Indicates turnover pressure and disruption ability. |
                | off_success_diff | Offensive success rate difference. | Represents consistency in gaining needed yardage. |
                | off_rushing_epa_diff | Rushing EPA per play difference. | Reveals the strength of the run game advantage. |
                | off_passing_epa_diff | Passing EPA per play difference. | Reveals the strength of the passing game advantage. |
                | recruiting_diff | Recruiting score/talent difference. | Proxy for roster talent depth and future upside. |
                | talent_diff | Overall talent rating difference. | General strength gap between home and away rosters. |
                | recruiting_rank_diff | Recruiting class rank difference. | Indicates relative talent quality by incoming classes. |
                | home_flag | Indicator for the home team. | Captures home-field advantage effects. |
                | conference_game | Indicator for conference matchup. | Accounts for rivalry / familiarity effects. |
                | rest_advantage | Home rest days minus away rest days. | Models fatigue or recovery advantage. |
                | turnover_margin_l5 | Last 5 games turnover margin difference. | Reflects recent ability to create and avoid turnovers. |
                | rushing_yards_diff_l5 | Last 5 games rushing yards difference. | Tracks recent ground-game dominance. |
                | pass_yards_diff_l5 | Last 5 games passing yards difference. | Tracks recent aerial-attack dominance. |
                | penalty_yards_diff_l5 | Last 5 games penalty yards difference. | Captures discipline and self-inflicted disadvantage. |
                | fpi_diff | Home minus away FPI rating difference. | Adds a consensus national power-rating signal. |
                | srs_diff | Home minus away SRS rating difference. | Adds a margin-adjusted, SOS-aware rating signal. |
                | returning_ppa_diff | Returning production percentage difference. | Captures roster continuity and experience. |
                | ppa_off_diff | Offensive PPA difference between teams. | Reflects opponent-adjusted offensive efficiency. |
                | ppa_def_diff | Defensive PPA difference between teams. | Reflects opponent-adjusted defensive efficiency. |
                | ppa_third_down_off_diff | Third-down offensive PPA difference. | Measures efficiency in critical third-down situations. |
                | ppa_third_down_def_diff | Third-down defensive PPA difference. | Measures ability to stop opponents on third downs. |
                | wepa_off_diff | Opponent-adjusted offensive EPA difference. | Accounts for schedule strength and tempo-adjusted offense. |
                | wepa_def_diff | Opponent-adjusted defensive EPA difference. | Accounts for schedule strength and tempo-adjusted defense. |
                | cfbd_pregame_wp_diff | CFBD pre-game home win probability minus 0.5. | Provides a consensus probability baseline for the matchup. |
                | coach_tenure_diff | Home coach tenure minus away coach tenure. | Models coaching experience and first-year coach risk. |
                | market_spread | Closing market spread, used as the consensus line. | Anchors the model to the betting market and line value. |
                """
            )
    else:
        st.info("Feature importance not available for this model type.")
else:
    st.info("Spread model not loaded.")

st.divider()

st.subheader("Feature Importance — Total Model")
total_mod = models.get("total")

if total_mod is not None:
    if HAS_XGB and isinstance(total_mod, xgb.Booster):
        raw_imp = total_mod.get_score(importance_type="gain")
        imp_df_total = (
            pd.DataFrame(
                [{"Feature": f"f{i}", "Importance": raw_imp.get(f"f{i}", 0)}
                 for i in range(len(TOTAL_FEATURES))]
            )
            .assign(Feature=lambda d: [
                TOTAL_FEATURES[int(r["Feature"][1:])] if int(r["Feature"][1:]) < len(TOTAL_FEATURES)
                else r["Feature"]
                for _, r in d.iterrows()
            ])
            .sort_values("Importance", ascending=True)
        )
    else:
        try:
            coefs = abs(total_mod.named_steps["ridge"].coef_)
            used_feats = [f for f in TOTAL_FEATURES if f in (df.columns if not df.empty else TOTAL_FEATURES)][:len(coefs)]
            imp_df_total = pd.DataFrame({"Feature": used_feats, "Importance": coefs}).sort_values("Importance")
        except Exception:
            imp_df_total = pd.DataFrame()

    if not imp_df_total.empty:
        fig_imp_total = go.Figure(go.Bar(
            x=imp_df_total["Importance"], y=imp_df_total["Feature"],
            orientation="h",
            marker_color="#D4001C",
        ))
        fig_imp_total.update_layout(
            xaxis_title="Importance",
            paper_bgcolor="#F7FBFF",
            plot_bgcolor="#F7FBFF",
            font=dict(color="#1A2B3C"),
        )
        st.plotly_chart(fig_imp_total, width="stretch")
        with st.expander("Total Model Feature Dictionary", expanded=False):
            st.markdown(
                """
                | Feature | Description | Why helpful |
                |---|---|---|
                | home_off_epa | Home offense EPA per play. | Measures home team's scoring efficiency. |
                | away_off_epa | Away offense EPA per play. | Measures away team's scoring efficiency. |
                | home_def_epa | Home defense EPA allowed per play. | Indicates how well the home defense limits opponents. |
                | away_def_epa | Away defense EPA allowed per play. | Indicates how well the away defense limits opponents. |
                | home_off_explosiveness | Home offensive explosiveness. | Captures big-play scoring upside at home. |
                | away_off_explosiveness | Away offensive explosiveness. | Captures big-play scoring upside on the road. |
                | home_off_rushing_epa | Home rushing EPA per play. | Measures home running game scoring value. |
                | away_off_rushing_epa | Away rushing EPA per play. | Measures away running game scoring value. |
                | home_off_passing_epa | Home passing EPA per play. | Measures home passing game scoring value. |
                | away_off_passing_epa | Away passing EPA per play. | Measures away passing game scoring value. |
                | home_flag | Indicator for the home team. | Captures home-field scoring advantage. |
                | rest_days_home | Days of rest for home team. | Models freshness and recovery effects. |
                | rest_days_away | Days of rest for away team. | Models fatigue and travel effects. |
                | market_total | Betting market total line. | Anchors the model to market scoring expectations. |
                | is_dome | Indoor/dome game flag. | Adjusts for weather-insulated scoring environments. |
                | temperature | Forecast temperature. | Cold games usually suppress scoring. |
                | wind_speed | Forecast wind speed. | High wind reduces passing efficiency and scoring. |
                | adverse_weather | Bad weather indicator. | Captures rain, snow, wind, or cold impact on scoring. |
                | high_wind | High-wind indicator. | Highlights games where kicking and passing suffer. |
                | high_altitude | High elevation venue flag. | Models altitude effects on scoring and endurance. |
                | is_primetime | Prime-time game flag. | Captures TV/prime-time scoring and officiating effects. |
                """
            )
    else:
        st.info("Feature importance not available for this model type.")
else:
    st.info("Total model not loaded.")

# ─────────────────── sample size summary ───────────────────────────────────────
st.divider()
st.subheader("Training Data Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Win model samples",    f"{win_m.get('n_samples', '—'):,}"    if win_m.get('n_samples') else "—")
c2.metric("Spread model samples", f"{spread_m.get('n_samples', '—'):,}" if spread_m.get('n_samples') else "—")
c3.metric("Total model samples",  f"{total_m.get('n_samples', '—'):,}"  if total_m.get('n_samples') else "—")

add_betting_oracle_footer()
