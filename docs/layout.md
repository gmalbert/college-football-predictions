# Site Layout & UX Roadmap

This document defines the multi-page Streamlit layout, page-by-page wireframes,
and the sidebar / navigation structure.

---

## 1. Page Architecture

Streamlit supports multi-page apps via a `pages/` directory.  The recommended
file structure:

```
college-football-predictions/
├── predictions.py               # 🏠 Home / landing page (entry point)
├── pages/
│   ├── 1_📊_Weekly_Predictions.py
│   ├── 2_💰_Value_Bets.py
│   ├── 3_🏟️_Team_Explorer.py
│   ├── 4_📈_Historical_Analysis.py
│   ├── 5_🎯_Model_Performance.py
│   └── 6_⚙️_Settings.py
├── utils/
│   ├── __init__.py
│   ├── cfbd_client.py           # CFBD API wrapper
│   ├── espn_client.py           # ESPN API wrapper
│   ├── feature_engine.py        # Feature engineering
│   ├── models.py                # Model loading & inference
│   └── ui_components.py         # Shared Streamlit components
├── data_files/
│   ├── logo.png
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── models/
├── docs/
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml             # (gitignored)
├── requirements.txt
└── README.md
```

---

## 2. Page-by-Page Wireframes

### Page 0: Home (`predictions.py`)

```
┌─────────────────────────────────────────────────────────┐
│                      [LOGO]                             │
│                                                         │
│  College Football Predictions & Betting                 │
│  ─────────────────────────────────────                  │
│  Brief site description, data sources, how it works     │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ This Week's │  │  Top Value   │  │  Model        │  │
│  │ Top Picks   │  │  Bets        │  │  Accuracy     │  │
│  │ (3 cards)   │  │  (3 cards)   │  │  (sparkline)  │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Page 1: Weekly Predictions (`1_📊_Weekly_Predictions.py`)

```
┌─────────────────────────────────────────────────────────┐
│ Week Selector: [◀ Week 5 ▶]    Season: [2025]          │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Game Card (repeating)                               │ │
│ │ ┌──────┐  vs  ┌──────┐                              │ │
│ │ │ Away │      │ Home │  Model Spread: -6.5          │ │
│ │ │ Logo │      │ Logo │  Book Spread:  -7.0          │ │
│ │ └──────┘      └──────┘  Win Prob: 72% 🟢            │ │
│ │                         O/U Model: 54.5 | Book: 52  │ │
│ │  ▸ Show detailed breakdown                          │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Filter: [Conference ▼] [Min Edge ▼] [Sort by ▼]        │
└─────────────────────────────────────────────────────────┘
```

### Page 2: Value Bets (`2_💰_Value_Bets.py`)

```
┌─────────────────────────────────────────────────────────┐
│ Value Bets — Games Where the Model Disagrees w/ Vegas   │
│                                                         │
│ Bet Type: [Spread] [Moneyline] [Total]                  │
│ Min Edge: [──●────] 2.0 points                          │
│                                                         │
│ ┌──────────────────────────────────────────────────────┐│
│ │ Table: sortable, filterable                          ││
│ │ Game | Model | Book | Edge | Confidence | Pick       ││
│ │ ───────────────────────────────────────────────────  ││
│ │ OSU@MICH | -3.5 | -7 | 3.5 | HIGH | Take Michigan  ││
│ │ UGA@LSU  | -1.0 | -3 | 2.0 | MED  | Take LSU       ││
│ └──────────────────────────────────────────────────────┘│
│                                                         │
│ Bankroll Simulator: [ starting $1000 ]  [ flat / kelly ]│
│ [Chart: cumulative P&L over season]                     │
└─────────────────────────────────────────────────────────┘
```

### Page 3: Team Explorer (`3_🏟️_Team_Explorer.py`)

```
┌─────────────────────────────────────────────────────────┐
│ Team: [Alabama ▼]   Season: [2025]                      │
│                                                         │
│ ┌──────────────────────┐  ┌───────────────────────────┐ │
│ │ Team Card            │  │ Elo History (line chart)  │ │
│ │ Record: 8-2          │  │ Shows Elo rating per week │ │
│ │ Conf: SEC            │  │ with season overlay       │ │
│ │ Ranking: #6          │  └───────────────────────────┘ │
│ │ Talent: 912          │                                │
│ └──────────────────────┘                                │
│                                                         │
│ ┌──────────────────────────────────────────────────────┐│
│ │ Advanced Stats Radar Chart                           ││
│ │ Off EPA | Def EPA | Havoc | Success Rate | Pace      ││
│ └──────────────────────────────────────────────────────┘│
│                                                         │
│ ┌──────────────────────────────────────────────────────┐│
│ │ Schedule & Results Table                             ││
│ │ Wk | Opponent | Result | Spread | ATS | Model Pred  ││
│ └──────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### Page 4: Historical Analysis (`4_📈_Historical_Analysis.py`)

```
┌─────────────────────────────────────────────────────────┐
│ Season Range: [2015] to [2025]                          │
│                                                         │
│ Tab: [Season Trends] [H2H Lookup] [Conference Power]   │
│                                                         │
│ Season Trends:                                          │
│  - ATS record by conference (bar chart)                 │
│  - Home-field advantage over time (line chart)          │
│  - Avg scoring by era (area chart)                      │
│                                                         │
│ H2H Lookup:                                             │
│  Team A: [Ohio State] vs Team B: [Michigan]             │
│  All-time record, last 10 meetings, ATS history         │
│                                                         │
│ Conference Power:                                       │
│  - Heatmap of non-conference records                    │
│  - Recruiting spending by conference                    │
└─────────────────────────────────────────────────────────┘
```

### Page 5: Model Performance (`5_🎯_Model_Performance.py`)

```
┌─────────────────────────────────────────────────────────┐
│ Model Dashboard                                         │
│                                                         │
│ ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │
│ │ Brier: 0.19│ │ ATS: 54.2% │ │ ROI: +3.2%        │   │
│ └────────────┘ └────────────┘ └────────────────────┘   │
│                                                         │
│ ┌──────────────────────────────────────────────────────┐│
│ │ Calibration Curve (predicted vs actual win %)        ││
│ └──────────────────────────────────────────────────────┘│
│                                                         │
│ ┌──────────────────────────────────────────────────────┐│
│ │ ATS Record by Week (bar chart)                       ││
│ └──────────────────────────────────────────────────────┘│
│                                                         │
│ ┌──────────────────────────────────────────────────────┐│
│ │ Feature Importance (horizontal bar chart)            ││
│ └──────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## 3. Sidebar Design

The sidebar is persistent across all pages:

```python
# utils/ui_components.py (sidebar fragment)
import streamlit as st

def render_sidebar():
    """Standard sidebar rendered on every page."""
    with st.sidebar:
        st.image("data_files/logo.png", use_container_width=True)
        st.divider()

        # Quick stats
        st.metric("Season ATS", "54-46", "+3.2% ROI")
        st.metric("Model Brier", "0.192", "-0.008 vs last week")
        st.divider()

        # Live scores (game days only)
        st.subheader("🔴 Live Scores")
        st.caption("No games in progress")

        st.divider()
        st.caption("Data: CFBD API · ESPN | Updated daily")
```

---

## 4. Streamlit Configuration

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#D4001C"       # crimson red — football feel
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1A1E2C"
textColor = "#FAFAFA"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false
```

---

## 5. Responsive Design Notes

- Use `st.columns()` for desktop multi-column layouts.
- All images use `use_container_width=True` for mobile scaling.
- Prefer `st.dataframe()` (scrollable) over `st.table()` for large datasets.
- Use `st.expander()` to keep pages scannable on mobile.
- Test on 375px (mobile), 768px (tablet), 1440px (desktop).

---

## 6. UI Component Library

Shared components to keep consistent across pages:

```python
"""utils/ui_components.py — Reusable Streamlit UI components."""
import streamlit as st
import pandas as pd


def game_card(
    home: str,
    away: str,
    model_spread: float,
    book_spread: float,
    win_prob: float,
    model_total: float | None = None,
    book_total: float | None = None,
) -> None:
    """Render a single game prediction card."""
    edge = abs(model_spread - book_spread)
    edge_color = "🟢" if edge >= 2 else "🟡" if edge >= 1 else "⚪"

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"### {away}")
        with col2:
            st.markdown("### @")
        with col3:
            st.markdown(f"### {home}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Prob", f"{win_prob:.0%}")
        c2.metric("Model Spread", f"{model_spread:+.1f}")
        c3.metric("Book Spread", f"{book_spread:+.1f}")
        c4.metric("Edge", f"{edge:.1f} {edge_color}")

        if model_total and book_total:
            t1, t2 = st.columns(2)
            t1.metric("Model Total", f"{model_total:.1f}")
            t2.metric("Book Total", f"{book_total:.1f}")

        st.divider()


def value_badge(edge: float) -> str:
    """Return a colored badge string based on edge size."""
    if edge >= 3:
        return "🔥 STRONG VALUE"
    elif edge >= 2:
        return "✅ VALUE"
    elif edge >= 1:
        return "⚠️ LEAN"
    return ""


def metric_row(metrics: dict[str, tuple[str, str]]) -> None:
    """
    Render a row of st.metric() calls.
    metrics: {"Label": ("value", "delta"), ...}
    """
    cols = st.columns(len(metrics))
    for col, (label, (value, delta)) in zip(cols, metrics.items()):
        col.metric(label, value, delta)
```

---

## 7. Implementation Phases

| Phase | Pages | Target |
|-------|-------|--------|
| **MVP** | Home + Weekly Predictions | Week 1 |
| **v0.2** | Value Bets + Team Explorer | Week 3 |
| **v0.3** | Historical Analysis | Week 5 |
| **v0.4** | Model Performance + Settings | Week 7 |
| **v1.0** | Live scores, polish, mobile QA | Week 9 |
