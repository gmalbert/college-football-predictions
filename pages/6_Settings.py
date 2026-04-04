"""pages/6_⚙️_Settings.py

Admin / configuration page:
  - API key status
  - Data pull controls (one button → 46 API calls total, cached)
  - Model training controls
  - Manual refresh for processed tables
"""
from __future__ import annotations

import os
import streamlit as st

from utils.ui_components import render_sidebar
from utils.storage import RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR
from utils.logger import get_logger
from footer import add_betting_oracle_footer

logger = get_logger(__name__)


render_sidebar()
st.title("⚙️ Settings")

# ── API key status ─────────────────────────────────────────────────────────────
st.subheader("🔑 API Configuration")

has_key = False
try:
    from utils.config import get_secret
    key = get_secret("cfbd", "api_key")
    has_key = bool(key)
except Exception:
    has_key = False

if has_key:
    st.success("✅ CFBD API key detected.")
else:
    st.error("❌ No CFBD API key found.")
    st.markdown(
        """
        **To add your API key**, create `.streamlit/secrets.toml`:
        ```toml
        [cfbd]
        api_key = "YOUR_KEY_HERE"
        ```
        Or set the environment variable `CFBD_API_KEY`.
        Get a free key at [collegefootballdata.com](https://collegefootballdata.com).
        """
    )

st.divider()

# ── data status overview ───────────────────────────────────────────────────────
st.subheader("📂 Data Status")

from utils.fetch_historical import HISTORICAL_YEARS

raw_files   = list(RAW_DIR.glob("*.json"))   if RAW_DIR.exists()   else []
proc_files  = list(PROCESSED_DIR.glob("*.parquet")) if PROCESSED_DIR.exists() else []
feat_files  = list(FEATURES_DIR.glob("*.parquet"))  if FEATURES_DIR.exists()  else []
model_files = list(MODELS_DIR.glob("*.joblib"))      if MODELS_DIR.exists()    else []

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw cache files",       len(raw_files),   help="data_files/raw/")
c2.metric("Processed Parquets",    len(proc_files),  help="data_files/processed/")
c3.metric("Feature tables",        len(feat_files),  help="data_files/features/")
c4.metric("Trained model files",   len(model_files), help="data_files/models/")

# Expected raw files count: 1 (teams) + 5 years × 9 endpoints = 46
total_expected = 1 + len(HISTORICAL_YEARS) * 9
st.progress(
    value=min(len(raw_files) / total_expected, 1.0),
    text=f"Raw data: {len(raw_files)} / {total_expected} expected files",
)

st.divider()

# ── data pull ──────────────────────────────────────────────────────────────────
st.subheader("📥 Pull Historical Data")
st.markdown(
    f"Pulls **{total_expected} API calls** for seasons "
    f"{HISTORICAL_YEARS[0]}–{HISTORICAL_YEARS[-1]}. "
    "Already-cached files are skipped automatically."
)

col1, col2 = st.columns([1, 3])
with col1:
    force_pull = st.checkbox("Force re-pull (ignore cache)", value=False)
with col2:
    st.caption("⚠️ Force re-pull will use all API calls again (~46 requests).")

if st.button("🚀 Pull Historical Data", disabled=not has_key):
    if not has_key:
        st.error("Add your CFBD API key first.")
    else:
        with st.spinner("Pulling data… this may take ~60 seconds on first run."):
            try:
                from utils.fetch_historical import run as fetch_run
                log_placeholder = st.empty()
                fetch_run(force=force_pull)
                st.success("✅ Data pull complete!")
                st.rerun()
            except Exception as exc:
                st.error(f"Data pull failed: {exc}")
                logger.exception("fetch_historical.run() failed")

# ── process raw → parquet ──────────────────────────────────────────────────────
if raw_files:
    st.divider()
    st.subheader("🔄 (Re)build Processed Tables")
    st.caption(
        "Use this if raw JSON cache is present but Parquet tables are missing "
        "or stale. Does not consume any API calls."
    )
    proc_force = st.checkbox("Force overwrite existing Parquets", value=False, key="proc_force")
    if st.button("🔧 Build Processed Tables"):
        with st.spinner("Processing…"):
            try:
                from utils.fetch_historical import build_processed_tables
                build_processed_tables(force=proc_force)
                st.success("✅ Processed tables rebuilt.")
                st.rerun()
            except Exception as exc:
                st.error(f"Processing failed: {exc}")

# ── feature matrix ─────────────────────────────────────────────────────────────
if proc_files:
    st.divider()
    st.subheader("🧮 Rebuild Feature Matrix")
    st.caption("Joins all processed Parquets into the game-level feature matrix.")
    feat_force = st.checkbox("Force overwrite", value=False, key="feat_force")
    if st.button("🧮 Build Feature Matrix"):
        with st.spinner("Building feature matrix…"):
            try:
                from utils.feature_engine import build_feature_matrix
                df = build_feature_matrix(force=feat_force)
                st.success(f"✅ Feature matrix: {len(df):,} rows.")
                st.rerun()
            except Exception as exc:
                st.error(f"Feature engineering failed: {exc}")

# ── model training ─────────────────────────────────────────────────────────────
if feat_files:
    st.divider()
    st.subheader("🤖 Train Models")
    st.markdown(
        "Trains three models on the feature matrix using time-series cross-validation:\n"
        "- **Win Probability** — XGBoost binary classifier (Logistic Regression fallback)\n"
        "- **Spread** — XGBoost regressor (Ridge fallback)\n"
        "- **Totals (O/U)** — Ridge regression\n"
    )
    train_force = st.checkbox("Force retrain (ignore existing models)", value=False, key="train_force")
    if st.button("🎯 Train Models"):
        with st.spinner("Training… this may take a few minutes on the full dataset."):
            try:
                from utils.models import train_all
                metrics = train_all(force=train_force)
                if metrics:
                    st.success("✅ Models trained successfully!")
                    ats = metrics.get("ats", {})
                    win = metrics.get("win_model", {})
                    spread = metrics.get("spread_model", {})
                    st.json({
                        "win_brier":    round(win.get("brier", 0), 4),
                        "spread_rmse":  round(spread.get("rmse", 0), 2),
                        "ats_win_pct":  f"{ats.get('pct', 0):.1%}",
                        "ats_record":   f"{ats.get('wins',0)}W-{ats.get('losses',0)}L",
                    })
                    st.rerun()
            except Exception as exc:
                st.error(f"Model training failed: {exc}")
                logger.exception("train_all() failed")

# ── data files browser ────────────────────────────────────────────────────────
st.divider()
with st.expander("📁 View data file details"):
    col_r, col_p, col_f = st.columns(3)
    with col_r:
        st.caption("**Raw JSON**")
        for f in sorted(raw_files):
            st.text(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
    with col_p:
        st.caption("**Processed Parquet**")
        for f in sorted(proc_files):
            st.text(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
    with col_f:
        st.caption("**Features / Models**")
        for f in sorted(feat_files + model_files):
            st.text(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

st.divider()
add_betting_oracle_footer()
