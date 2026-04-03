import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="College Football Predictions",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------
logo_path = Path(__file__).parent / "data_files" / "logo.png"
if logo_path.exists():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(str(logo_path), use_container_width=True)

# ---------------------------------------------------------------------------
# Title & description
# ---------------------------------------------------------------------------
st.markdown(
    """
    # 🏈 College Football Predictions & Betting

    Welcome to **College Football Predictions** — a data-driven platform built
    to help you make smarter bets on college football games.

    ---

    ### What This Site Does

    * **Game Predictions** — Machine-learning models trained on historical and
      live data produce win-probability, spread, and over/under forecasts for
      every FBS matchup.
    * **Betting Value Finder** — Compares model outputs against current
      sportsbook lines to surface edges and +EV opportunities.
    * **Team & Player Dashboards** — Explore advanced stats, trends, and
      head-to-head histories powered by the
      [College Football Data API](https://api.collegefootballdata.com/) and the
      ESPN API.
    * **Historical Analysis** — Dive into decade-plus archives of scores,
      rankings, recruiting, and betting-line movement.
    * **Live Game Tracker** — Real-time score updates in the sidebar during
      game days so you never miss a beat.

    ---

    ### Data Sources

    | Source | What It Provides |
    |--------|-----------------|
    | **College Football Data API** | Scores, stats, rankings, recruiting, betting lines, play-by-play, EPA, team records, conferences |
    | **ESPN API** | Live scores, schedules, team rosters, game summaries, odds |

    ---

    ### How It Works

    1. **Collect** — Automated pipelines pull the latest data every day (and
       live on game days).
    2. **Transform** — Raw data is cleaned, feature-engineered, and stored for
       model consumption.
    3. **Predict** — An ensemble of models (gradient-boosted trees, logistic
       regression, and neural nets) produces forecasts.
    4. **Present** — Results are displayed here with clear visuals, confidence
       intervals, and historical accuracy tracking.

    ---

    > *Use the sidebar to navigate between pages — Predictions, Value Bets,
    > Team Explorer, and more are coming soon!*
    """
)

# ---------------------------------------------------------------------------
# Sidebar — placeholder for future navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Navigation")
    st.markdown(
        """
        - 🏠 **Home** *(you are here)*
        - 📊 Predictions *(coming soon)*
        - 💰 Value Bets *(coming soon)*
        - 🏟️ Team Explorer *(coming soon)*
        - 📈 Historical Analysis *(coming soon)*
        - ⚙️ Model Performance *(coming soon)*
        """
    )
    st.divider()
    st.caption("Data refreshed daily · Models retrained weekly")
