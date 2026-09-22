"""Preseason Outlook — mirrors ``pages/8_Preseason_Outlook.py``."""
from __future__ import annotations

import pandas as pd
import plotly.express as px

from api.charts import figure_json
from api.data import parquet
from api.jsonutil import records


def _load(name: str) -> pd.DataFrame:
    try:
        return parquet(name)
    except FileNotFoundError:
        return pd.DataFrame()


def _returning_production(season: int, ret_df: pd.DataFrame) -> dict:
    heading = f"{season} — Returning Production Leaderboard"
    if ret_df.empty:
        return {
            "id": "returning",
            "label": "📊 Returning Production",
            "heading": heading,
            "info": "No returning production data. Run the data pipeline first.",
        }

    df = ret_df[ret_df["season"] == season].copy()
    if df.empty:
        return {
            "id": "returning",
            "label": "📊 Returning Production",
            "heading": heading,
            "warning": f"No returning production data for {season}.",
        }

    pct_columns = [
        "percent_ppa", "percent_passing_ppa", "percent_receiving_ppa", "percent_rushing_ppa",
    ]
    for column in pct_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            if df[column].median() < 2:
                df[column] = df[column] * 100

    df = df.sort_values("percent_ppa", ascending=False)

    col_headers = {
        "team": "Team",
        "conference": "Conference",
        "percent_ppa": "Total PPA Returning %",
        "percent_passing_ppa": "Passing PPA Returning %",
        "percent_rushing_ppa": "Rushing PPA Returning %",
        "percent_receiving_ppa": "Receiving PPA Returning %",
    }
    show_cols = [
        column for column in
        ["team", "conference", "percent_ppa", "percent_passing_ppa",
         "percent_rushing_ppa", "percent_receiving_ppa"]
        if column in df.columns
    ]
    display_df = df[show_cols].head(130).rename(columns=col_headers)

    progress_columns = {
        label: {"min": 0, "max": 100, "format": "%.1f%%"}
        for key, label in col_headers.items()
        if key in show_cols and label.endswith("%")
    }

    chart = None
    if "percent_ppa" in df.columns and "team" in df.columns:
        top30 = df[show_cols].head(30).copy()
        fig = px.bar(
            top30, x="team", y="percent_ppa",
            color="conference" if "conference" in top30.columns else None,
            title=f"{season} — Top 30 Teams by Returning PPA %",
            labels={"percent_ppa": "% of PPA Returning", "team": "Team"},
        )
        fig.update_layout(
            xaxis_tickangle=-45, height=420,
            paper_bgcolor="#F7FBFF", plot_bgcolor="#F7FBFF",
            font=dict(color="#1A2B3C"),
        )
        chart = figure_json(fig)

    return {
        "id": "returning",
        "label": "📊 Returning Production",
        "heading": heading,
        "table": {
            "columns": [str(column) for column in display_df.columns],
            "rows": records(display_df),
            "height": 480,
            "progress_columns": progress_columns,
        },
        "figure": chart,
    }


def _transfer_portal(season: int, portal_df: pd.DataFrame) -> dict:
    heading = f"{season} — Transfer Portal Net Impact"
    base = {"id": "portal", "label": "🔄 Transfer Portal", "heading": heading}

    if portal_df.empty:
        return {**base, "info": "No transfer portal data. Run the data pipeline first."}

    pf = portal_df[portal_df["season"] == season].copy()
    if pf.empty:
        return {**base, "warning": f"No portal data for {season}."}

    pf["rating"] = pd.to_numeric(pf["rating"], errors="coerce").fillna(0)

    gains = (
        pf.groupby("destination")["rating"]
        .agg(gains_sum="sum", gains_count="count")
        .reset_index()
        .rename(columns={"destination": "team"})
    )
    losses = (
        pf.groupby("origin")["rating"]
        .agg(losses_sum="sum", losses_count="count")
        .reset_index()
        .rename(columns={"origin": "team"})
    )
    merged = gains.merge(losses, on="team", how="outer").fillna(0)
    merged["net_rating"] = merged["gains_sum"] - merged["losses_sum"]
    merged["net_count"] = merged["gains_count"] - merged["losses_count"]
    merged = merged.sort_values("net_rating", ascending=False)

    gain_cols = {
        "team": "Team",
        "gains_sum": "Gains Rating",
        "gains_count": "Gain Count",
        "losses_sum": "Losses Rating",
        "net_rating": "Net Rating",
    }
    columns = ["team", "gains_sum", "gains_count", "losses_sum", "net_rating"]
    column_config = {
        "Gains Rating": {"format": "%.2f"},
        "Gain Count": {"format": "%.2f"},
        "Losses Rating": {"format": "%.2f"},
        "Net Rating": {"format": "%.2f"},
    }

    gainers = merged[columns].head(20).rename(columns=gain_cols)
    losers = merged[columns].tail(20).rename(columns=gain_cols)

    fig = px.bar(
        merged.head(30), x="team", y="net_rating",
        color="net_rating", color_continuous_scale="RdYlGn",
        title=f"{season} — Top 30 Teams by Transfer Portal Net Rating",
        labels={"net_rating": "Net Rating (Gains − Losses)", "team": "Team"},
    )
    fig.update_layout(
        xaxis_tickangle=-45, height=420, coloraxis_showscale=False,
        paper_bgcolor="#F7FBFF", plot_bgcolor="#F7FBFF",
        font=dict(color="#1A2B3C"),
    )

    return {
        **base,
        "gainers_heading": "**Top Gainers (by recruit rating sum)**",
        "losers_heading": "**Biggest Losers**",
        "gainers": {
            "columns": [str(column) for column in gainers.columns],
            "rows": records(gainers),
            "column_config": column_config,
        },
        "losers": {
            "columns": [str(column) for column in losers.columns],
            "rows": records(losers),
            "column_config": column_config,
        },
        "figure": figure_json(fig),
    }


def _efficiency_quadrant(season: int, adv_df: pd.DataFrame, conference: str) -> dict:
    heading = f"{season} — Team Efficiency Quadrant"
    base = {
        "id": "efficiency",
        "label": "⚡ Team Efficiency Quadrant",
        "heading": heading,
        "caption": "Top-right = elite offense AND elite defense. Source: CFBD advanced stats.",
    }

    if adv_df.empty:
        return {**base, "info": "No advanced stats data. Run the data pipeline first."}

    df3 = adv_df[adv_df["season"] == season].copy()
    if df3.empty:
        return {**base, "warning": f"No advanced stats for {season}."}

    for column in ["off_epa", "def_epa"]:
        if column in df3.columns:
            df3[column] = pd.to_numeric(df3[column], errors="coerce")

    if "off_epa" not in df3.columns or "def_epa" not in df3.columns:
        return {**base, "info": "off_epa / def_epa columns not found in advanced stats."}

    conferences = []
    selected_conf = "All"
    if "conference" in df3.columns:
        conferences = sorted(df3["conference"].dropna().unique().tolist())
        selected_conf = conference if conference in conferences else "All"
        if selected_conf != "All":
            df3 = df3[df3["conference"] == selected_conf]

    df3 = df3.dropna(subset=["off_epa", "def_epa"])
    if df3.empty:
        return {**base, "conferences": conferences, "conference": selected_conf,
                "warning": f"No data for {selected_conf} conference in {season}."}

    fig3 = px.scatter(
        df3, x="off_epa", y="def_epa", text="team",
        color="conference" if "conference" in df3.columns else None,
        title=f"{season} — Offense EPA/Play vs Defense EPA/Play",
        labels={
            "off_epa": "Offensive EPA/Play (higher = better)",
            "def_epa": "Defensive EPA/Play (lower = better)",
        },
        height=580,
    )
    fig3.update_traces(textposition="top center", marker_size=9)
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig3.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
    x_max = float(df3["off_epa"].max()) * 0.8
    y_min = float(df3["def_epa"].min()) * 0.8
    x_min = float(df3["off_epa"].min()) * 0.8
    y_max = float(df3["def_epa"].max()) * 0.8
    for txt, xx, yy in [
        ("ELITE", x_max, y_min),
        ("OFFENSE ONLY", x_max, y_max),
        ("DEFENSE ONLY", x_min, y_min),
        ("REBUILDING", x_min, y_max),
    ]:
        fig3.add_annotation(
            x=xx, y=yy, text=txt, showarrow=False,
            font=dict(size=13, color="rgba(100,100,100,0.6)"),
        )
    fig3.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="#F7FBFF", plot_bgcolor="#F7FBFF",
        font=dict(color="#1A2B3C"),
    )

    return {
        **base,
        "conferences": ["All"] + conferences,
        "conference": selected_conf,
        "figure": figure_json(fig3),
    }


def build_preseason(season: int | None = None, conference: str = "All") -> dict:
    """Return the Preseason Outlook payload."""
    ret_df = _load("returning_production")
    if ret_df.empty:
        avail_seasons = list(range(2025, 2020, -1))
    else:
        avail_seasons = sorted(
            ret_df["season"].dropna().unique().tolist(), reverse=True
        )

    season = int(avail_seasons[0]) if season is None else int(season)
    if season not in avail_seasons:
        season = int(avail_seasons[0])

    adv_df = _load("advanced_stats")
    portal_df = _load("transfer_portal")

    return {
        "page": "preseason",
        "title": "🏈 Preseason Outlook",
        "caption": (
            "Returning production and transfer portal data — "
            "critical context for early-season predictions."
        ),
        "sidebar": {
            "heading": "Filters",
            "seasons": [int(value) for value in avail_seasons],
            "season": season,
        },
        "tabs": [
            _returning_production(season, ret_df),
            _transfer_portal(season, portal_df),
            _efficiency_quadrant(season, adv_df, conference),
        ],
        # The Streamlit page omits the footer here (documented as U4 in
        # docs/UI_UX_ENHANCEMENTS.md) — parity is intentional.
        "footer": False,
    }
