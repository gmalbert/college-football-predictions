import streamlit as st
from pathlib import Path
import pandas as pd
import os
import base64

from footer import add_betting_oracle_footer
from utils.ui_components import render_sidebar

# ---------------------------------------------------------------------------
# Page configuration — must be top-level, first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tailgate Edge - College Football Predictions",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Home page function (called by st.navigation)
# ---------------------------------------------------------------------------
def home_page():
    render_sidebar()

    # ── Logo & title ───────────────────────────────────────────────────────────
    logo_path = Path(__file__).parent / "data_files" / "logo.png"
    if logo_path.exists():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(str(logo_path), width=120)
        with col2:
            st.markdown(
                """
                # 🏈 College Football Predictions
                ---
                """
            )
    else:
        st.markdown(
            """
            # 🏈 College Football Predictions
            ---
            """
        )

    # ── Live summary cards ────────────────────────────────────────────────────
    from utils.storage import load_parquet
    from utils.release import load_current_release
    try:
        from utils.models import load_metrics, models_trained, predict_batch
        model_runtime_available = True
    except Exception:
        # Keep the read-only dashboard available when a local Windows policy
        # blocks native ML extensions. The CI/production runtime still loads them.
        model_runtime_available = False

        def load_metrics():
            import json
            metrics_path = Path(__file__).parent / "data_files" / "models" / "model_metrics.json"
            return json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

        def models_trained():
            return False

        def predict_batch(frame):
            return frame

    @st.cache_data(ttl=3600)
    def _load_summary():
        try:
            df = load_parquet("feature_matrix", layer="features")
            if "home_margin" in df.columns:
                df = df[pd.to_numeric(df["home_margin"], errors="coerce").isna()].copy()
            elif {"home_score", "away_score"}.issubset(df.columns):
                df = df[df["home_score"].isna() | df["away_score"].isna()].copy()
            if "start_date" in df.columns:
                starts = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
                now = pd.Timestamp.now(tz="UTC")
                df = df[starts.isna() | (starts >= now - pd.Timedelta(hours=6))].copy()
            if models_trained() and not df.empty:
                df = predict_batch(df)
            return df
        except FileNotFoundError:
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def _metrics():
        return load_metrics()

    df_all   = _load_summary()
    metrics  = _metrics()
    ats_m    = metrics.get("ats", {})
    win_m    = metrics.get("win_model", {})
    spread_m = metrics.get("spread_model", {})
    release  = metrics.get("release_decision", {})
    release_metadata = load_current_release()

    if metrics and release.get("decision") != "promote":
        failed = ", ".join(release.get("failed_gates", [])) or "release gates unavailable"
        st.warning(
            f"Model release status: HOLD ({failed}). Forecasts are shown for research; "
            "automated bet export is disabled."
        )
    if not model_runtime_available:
        st.info("Native model inference is unavailable in this local runtime; saved evaluation evidence remains available.")
    if release_metadata:
        st.caption(
            f"Artifact release {release_metadata.get('release_id', 'unknown')[:12]} · "
            f"{release_metadata.get('generated_at', 'unknown')}"
        )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Upcoming Model Deltas")
        if not df_all.empty and "predicted_spread" in df_all.columns and "market_spread" in df_all.columns:
            top = (
                df_all
                .assign(edge=lambda d: (d["predicted_spread"] + d["market_spread"]).abs())
                .dropna(subset=["predicted_spread", "market_spread"])
                .nlargest(3, "edge")
            )
            for _, row in top.iterrows():
                ms = row.get("predicted_spread", float("nan"))
                bs = row.get("market_spread",    float("nan"))
                wp = row.get("win_prob",         float("nan"))
                if pd.notna(ms) and pd.notna(bs):
                    edge = abs(ms + bs)
                    st.markdown(
                        f"**{row['away_team']} @ {row['home_team']}**  \n"
                        f"Wk {int(row['week'])} · Edge **{edge:.1f} pts** · "
                        f"Win prob {wp:.0%}" if pd.notna(wp) else
                        f"Wk {int(row['week'])} · Edge **{edge:.1f} pts**"
                    )
            if top.empty:
                st.caption("No upcoming games with both model and market lines are available.")
        else:
            st.caption("No data yet. Go to ⚙️ Settings to pull historical data.")

    with col_b:
        st.subheader("📐 Model Accuracy")
        if metrics:
            st.metric("Brier Score",   f"{win_m.get('brier', 0):.4f}",   help="Lower is better; < 0.20 is solid")
            st.metric("Spread RMSE",   f"{spread_m.get('rmse', 0):.2f} pts")
            st.metric("OOS ATS Win %", f"{ats_m.get('pct', 0):.1%}",    help="Walk-forward only; 52.4% breaks even at -110")
            st.metric("ATS Record",    f"{ats_m.get('wins',0)}‑{ats_m.get('losses',0)}")
        else:
            st.caption("Models not yet trained — go to ⚙️ Settings → Train Models.")

    with col_c:
        st.subheader("📊 Dataset")
        if not df_all.empty:
            seasons = sorted(df_all["season"].dropna().unique())
            n_games = len(df_all)
            n_teams = len(set(df_all["home_team"].dropna().tolist() + df_all["away_team"].dropna().tolist()))
            st.metric("Games",   f"{n_games:,}")
            st.metric("Teams",   f"{n_teams:,}")
            st.metric("Seasons", f"{seasons[0]}–{seasons[-1]}")
            st.metric("Model",   "XGBoost + Ridge" if models_trained() else "Not trained")
        else:
            st.caption("No data loaded. Go to ⚙️ Settings to get started.")

    # st.divider()

    # # ── How it works ──────────────────────────────────────────────────────────
    # st.markdown(
    #     """
    #     ### How It Works

    #     | Step | What Happens |
    #     |------|-------------|
    #     | 1. **Collect** | ~46 CFBD API calls pull 5 seasons (2021–2025) of games, lines, EPA stats, Elo, SP+, recruiting, and talent data — cached locally so re-runs are instant. |
    #     | 2. **Transform** | Raw JSON → Parquet tables → joined feature matrix with 20+ per-game features (Elo diff, SP+ diff, EPA differentials, recruiting talent, home-field, betting lines). |
    #     | 3. **Predict** | Time-series cross-validated XGBoost models predict win probability, point spread, and over/under for every game. |
    #     | 4. **Surface edges** | Model lines are compared to sportsbook lines; games with ≥ 2 pt spread edge or ≥ 2.5 pt total edge are flagged as value bets. |
    #     | 5. **Present** | Results displayed with confidence tiers, calibration curves, ATS tracking, and a bankroll simulator. |

    #     ---

    #     ### Data Sources

    #     | Source | What It Provides |
    #     |--------|-----------------|
    #     | **[College Football Data API](https://api.collegefootballdata.com/)** | Scores, advanced stats (EPA, success rate, PPA), SP+ ratings, Elo ratings, betting lines, recruiting, talent composite |
    #     | **ESPN API** | Live scores, team rosters, current rankings |

    #     ---
    #     """
    # )

    add_betting_oracle_footer()


# ---------------------------------------------------------------------------
# Navigation — st.set_page_config() above already ensures we're inside
# Streamlit, so no runtime guard is needed here.
# ---------------------------------------------------------------------------
_is_cloud = bool(
    os.environ.get("IS_STREAMLIT_CLOUD")
    or os.environ.get("STREAMLIT_SHARING_MODE")
    or os.path.exists("/mount/src")          # Streamlit Community Cloud mounts here
)

nav_sections: dict = {
    "": [
        st.Page(home_page, title="Home", icon="🏈", default=True),
    ],
    "Analysis": [
        st.Page("pages/1_Weekly_Predictions.py",  title="Weekly Predictions",  icon="📊"),
        st.Page("pages/2_Value_Bets.py",           title="Value Bets",           icon="💰"),
        st.Page("pages/3_Team_Explorer.py",        title="Team Explorer",        icon="🏟️"),
        st.Page("pages/4_Historical_Analysis.py",  title="Historical Analysis",  icon="📈"),
        st.Page("pages/5_Model_Performance.py",    title="Model Performance",    icon="🎯"),
        st.Page("pages/7_Win_Probability.py",      title="Win Probability",      icon="📉"),
        st.Page("pages/8_Preseason_Outlook.py",   title="Preseason Outlook",   icon="🔮"),
        st.Page("pages/9_Data_Quality.py",        title="Data & Model Quality", icon="🛡️"),
    ],
}
if not _is_cloud:
    nav_sections["Config"] = [
        st.Page("pages/6_Settings.py", title="Settings", icon="⚙️"),
    ]

pg = st.navigation(nav_sections)

# Hide hamburger / manage-app buttons on Streamlit Cloud
if _is_cloud:
    st.markdown(
        """
        <style>
        [data-testid="main-menu-button"]  { display: none !important; }
        [data-testid="manage-app-button"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

pg.run()
