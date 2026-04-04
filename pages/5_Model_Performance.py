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
from utils.feature_engine import WIN_FEATURES, SPREAD_FEATURES
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
        )
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.info("Feature importance not available for this model type.")
else:
    st.info("Spread model not loaded.")

# ─────────────────── sample size summary ───────────────────────────────────────
st.divider()
st.subheader("Training Data Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Win model samples",    f"{win_m.get('n_samples', '—'):,}"    if win_m.get('n_samples') else "—")
c2.metric("Spread model samples", f"{spread_m.get('n_samples', '—'):,}" if spread_m.get('n_samples') else "—")
c3.metric("Total model samples",  f"{total_m.get('n_samples', '—'):,}"  if total_m.get('n_samples') else "—")

add_betting_oracle_footer()
