"""Team Explorer — mirrors ``pages/3_Team_Explorer.py``."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.charts import figure_json
from api.columns import FEATURE_MATRIX_COLUMNS
from api.data import parquet
from api.jsonutil import records
from utils.models import models_trained, predict_for_display


def load_all() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, layer in [
        ("feature_matrix", "features"),
        ("elo_ratings", "processed"),
        ("advanced_stats", "processed"),
        ("ratings", "processed"),
    ]:
        try:
            out[name] = parquet(
                name,
                layer=layer,
                columns=FEATURE_MATRIX_COLUMNS if name == "feature_matrix" else None,
            )
        except FileNotFoundError:
            out[name] = pd.DataFrame()
    return out


def _elo_figure(team: str, elo_df: pd.DataFrame) -> dict | None:
    if elo_df.empty:
        return None
    elo_team = elo_df[elo_df["team"] == team].sort_values("season")
    if elo_team.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=elo_team["season"], y=elo_team["elo"],
            mode="lines+markers",
            line=dict(color="#D4001C", width=2),
            marker=dict(size=7),
            name="Elo",
        )
    )
    fig.add_hline(
        y=1500, line_dash="dot", line_color="gray", annotation_text="Average (1500)"
    )
    fig.update_layout(
        title=f"{team} End-of-Season Elo Rating",
        xaxis_title="Season",
        yaxis_title="Elo",
        paper_bgcolor="#F7FBFF",
        plot_bgcolor="#F7FBFF",
        font=dict(color="#1A2B3C"),
    )
    return figure_json(fig)


RADAR_METRICS = {
    "Off EPA": "off_epa",
    "Off Success": "off_success_rate",
    "Off Explosive": "off_explosiveness",
    "Def EPA (inv)": "def_epa",
    "Def Success (inv)": "def_success_rate",
}


def _radar_figure(team: str, season: int, adv: pd.DataFrame) -> dict | None:
    if adv.empty:
        return None
    adv_row = adv[(adv["team"] == team) & (adv["season"] == season)]
    if adv_row.empty:
        return None
    row = adv_row.iloc[0]
    adv_season = adv[adv["season"] == season]
    radar_vals: list[float] = []
    radar_cats: list[str] = []
    for label, column in RADAR_METRICS.items():
        if column in adv_season.columns and pd.notna(row.get(column)):
            series = adv_season[column].dropna()
            val = float(row[column])
            pct = (series < val).mean()
            if "inv" in label.lower():
                pct = 1 - pct
            radar_vals.append(round(pct * 100, 1))
            radar_cats.append(label)
    if not radar_vals:
        return None
    fig = go.Figure(
        go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself",
            fillcolor="rgba(212,0,28,0.25)",
            line=dict(color="#D4001C"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100])),
        title=f"{team} Advanced Stats Percentile Rank — {season}",
        paper_bgcolor="#F7FBFF",
        font=dict(color="#1A2B3C"),
    )
    return figure_json(fig)


def _quadrant_figure(team: str, season: int, adv_all: pd.DataFrame) -> dict | None:
    if adv_all.empty:
        return None
    df_quad = adv_all[adv_all["season"] == season].copy()
    for column in ["off_epa", "def_epa"]:
        if column in df_quad.columns:
            df_quad[column] = pd.to_numeric(df_quad[column], errors="coerce")
    if "off_epa" in df_quad.columns and "def_epa" in df_quad.columns:
        df_quad = df_quad.dropna(subset=["off_epa", "def_epa"])
    else:
        df_quad = pd.DataFrame()
    if df_quad.empty:
        return None

    df_quad["_highlight"] = df_quad["team"] == team
    fig = px.scatter(
        df_quad,
        x="off_epa",
        y="def_epa",
        text="team",
        color="_highlight",
        color_discrete_map={True: "#D4001C", False: "#9ABBE0"},
        title=f"{season} — Team Efficiency Quadrant",
        labels={
            "off_epa": "Offensive EPA/Play (higher = better)",
            "def_epa": "Defensive EPA/Play (lower = better)",
        },
        height=500,
    )
    fig.update_traces(textposition="top center", marker_size=8)
    fig.update_traces(selector=dict(name="True"), marker_size=14, marker_symbol="star")
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
    x_hi = float(df_quad["off_epa"].quantile(0.88))
    y_lo = float(df_quad["def_epa"].quantile(0.12))
    x_lo = float(df_quad["off_epa"].quantile(0.12))
    y_hi = float(df_quad["def_epa"].quantile(0.88))
    for ann_txt, xx, yy in [
        ("ELITE", x_hi, y_lo),
        ("OFF ONLY", x_hi, y_hi),
        ("DEF ONLY", x_lo, y_lo),
        ("REBUILDING", x_lo, y_hi),
    ]:
        fig.add_annotation(
            x=xx, y=yy, text=ann_txt, showarrow=False,
            font=dict(size=12, color="rgba(100,100,100,0.55)"),
        )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="#F7FBFF",
        plot_bgcolor="#F7FBFF",
        font=dict(color="#1A2B3C"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return figure_json(fig)


def _schedule_rows(team: str, df_team: pd.DataFrame) -> list[dict]:
    if models_trained():
        df_team = predict_for_display(df_team)

    rows = []
    for _, g in df_team.iterrows():
        is_home = g["home_team"] == team
        opponent = g["away_team"] if is_home else g["home_team"]
        hs, as_ = g.get("home_score"), g.get("away_score")

        if pd.notna(hs) and pd.notna(as_):
            team_score = int(hs if is_home else as_)
            opp_score = int(as_ if is_home else hs)
            result = "W" if team_score > opp_score else "L"
            margin = team_score - opp_score
        else:
            team_score = opp_score = result = margin = None

        book_spread = g.get("market_spread")
        if pd.notna(book_spread) and pd.notna(margin):
            team_spread = book_spread if is_home else -book_spread
            settled = margin + team_spread
            ats = "✅" if settled > 0 else ("➡️" if settled == 0 else "❌")
        else:
            ats = "—"

        ms = g.get("predicted_spread")
        rows.append(
            {
                "Wk": int(g.get("week", 0)),
                "Opponent": opponent,
                "H/A": "H" if is_home else "A",
                "Result": f"{result} {team_score}‑{opp_score}" if result else "—",
                "Margin": f"{margin:+d}" if margin is not None else "—",
                "Book": f"{book_spread:+.1f}" if pd.notna(book_spread) else "—",
                "ATS": ats,
                "Model": f"{ms:+.1f}" if pd.notna(ms) else "—",
            }
        )
    return rows


def build_team_explorer(team: str | None = None, season: int | None = None) -> dict:
    """Return the Team Explorer payload."""
    data = load_all()
    fm = data["feature_matrix"]

    base = {"page": "team_explorer", "title": "🏟️ Team Explorer"}

    if fm.empty:
        return {**base, "warnings": ["No team data is currently published."], "stopped": True}

    all_teams = sorted(
        set(fm["home_team"].dropna().tolist() + fm["away_team"].dropna().tolist())
    )
    seasons = sorted(fm["season"].dropna().unique(), reverse=True)

    team = all_teams[all_teams.index("Alabama")] if team is None and "Alabama" in all_teams else team
    if team is None or team not in all_teams:
        team = all_teams[0]
    season = int(seasons[0]) if season is None else int(season)
    if season not in seasons:
        season = int(seasons[0])

    mask = ((fm["home_team"] == team) | (fm["away_team"] == team)) & (fm["season"] == season)
    df_team = fm[mask].copy().sort_values("week")

    home_games = df_team[df_team["home_team"] == team]
    away_games = df_team[df_team["away_team"] == team]

    wins = int((home_games["home_margin"] > 0).sum() + (away_games["home_margin"] < 0).sum())
    losses = len(df_team) - wins

    rat = data["ratings"]
    rat_row = rat[(rat["team"] == team) & (rat["season"] == season)]
    sp_plus = rat_row["sp_overall"].values[0] if not rat_row.empty else None
    talent = rat_row["talent"].values[0] if not rat_row.empty else None
    conf_val = df_team["home_conference"].dropna().mode()
    conference = conf_val.iloc[0] if not conf_val.empty else "—"

    elo_figure = _elo_figure(team, data["elo_ratings"])
    radar_figure = _radar_figure(team, season, data["advanced_stats"])
    quadrant_figure = _quadrant_figure(team, season, data["advanced_stats"])
    schedule_rows = _schedule_rows(team, df_team)

    return {
        **base,
        "teams": all_teams,
        "seasons": [int(value) for value in seasons],
        "selection": {"team": team, "season": season},
        "heading": team,
        "metrics": [
            {"label": "Record", "value": f"{wins}‑{losses}", "delta": None, "help": None},
            {"label": "Conference", "value": str(conference), "delta": None, "help": None},
            {"label": "SP+", "value": f"{sp_plus:.1f}" if sp_plus else "—", "delta": None, "help": None},
            {"label": "Talent", "value": f"{talent:.0f}" if talent else "—", "delta": None, "help": None},
            {"label": "Games", "value": str(len(df_team)), "delta": None, "help": None},
        ],
        "elo": {
            "figure": elo_figure,
            "info": None if elo_figure else (
                "Elo history not available for this team."
                if not data["elo_ratings"].empty else "Elo data not loaded."
            ),
        },
        "radar": {"figure": radar_figure},
        "quadrant": {
            "heading": "Team Efficiency Quadrant",
            "caption": (
                "Offense EPA/play vs Defense EPA/play across all teams. "
                "Selected team highlighted in red."
            ),
            "figure": quadrant_figure,
        },
        "schedule": {
            "heading": "Schedule & Results",
            "columns": ["Wk", "Opponent", "H/A", "Result", "Margin", "Book", "ATS", "Model"],
            "rows": records(pd.DataFrame(schedule_rows).reset_index(drop=True))
            if schedule_rows else [],
            "info": None if schedule_rows else "No schedule data for this team and season.",
        },
        "footer": True,
    }
