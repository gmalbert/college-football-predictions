"""Historical Analysis — mirrors ``pages/4_Historical_Analysis.py``."""
from __future__ import annotations

import pandas as pd
import plotly.express as px

from api.charts import figure_json
from api.data import parquet
from api.jsonutil import records


def load_data() -> pd.DataFrame:
    try:
        fm = parquet("feature_matrix", layer="features").drop_duplicates("game_id")
        try:
            backtest = parquet("model_backtest", layer="features")
            oos = backtest[["game_id", "predicted_spread_oos"]].drop_duplicates("game_id")
            fm = fm.merge(oos, on="game_id", how="left", validate="one_to_one")
        except FileNotFoundError:
            pass
        return fm
    except FileNotFoundError:
        return pd.DataFrame()


def _season_trends(df_slice: pd.DataFrame) -> dict:
    scoring = (
        df_slice.groupby("season")
        .agg(avg_total=("total_points", "mean"), avg_margin=("home_margin", "mean"))
        .reset_index()
    )

    fig_score = px.line(
        scoring, x="season", y="avg_total",
        title="Average Total Points per Game by Season",
        markers=True, color_discrete_sequence=["#D4001C"],
    )
    fig_score.update_layout(
        paper_bgcolor="#F7FBFF", plot_bgcolor="#F7FBFF", font=dict(color="#1A2B3C")
    )

    fig_hfa = px.bar(
        scoring, x="season", y="avg_margin",
        title="Average Home Margin by Season (positive = home advantage)",
        color_discrete_sequence=["#D4001C"],
    )
    fig_hfa.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_hfa.update_layout(
        paper_bgcolor="#F7FBFF", plot_bgcolor="#F7FBFF", font=dict(color="#1A2B3C")
    )

    ats_table = None
    if "predicted_spread_oos" in df_slice.columns and "market_spread" in df_slice.columns:
        ats_slice = df_slice.dropna(
            subset=["predicted_spread_oos", "market_spread", "home_margin"]
        ).copy()
        ats_slice = ats_slice[(ats_slice["home_margin"] + ats_slice["market_spread"]) != 0]
        ats_slice["pred_covers"] = ats_slice["home_margin"] > -ats_slice["market_spread"]
        ats_slice["model_picked_home"] = (
            ats_slice["predicted_spread_oos"] > -ats_slice["market_spread"]
        )
        ats_slice["ats_correct"] = ats_slice["pred_covers"] == ats_slice["model_picked_home"]

        ats_by_conf = (
            ats_slice.dropna(subset=["home_conference", "ats_correct"])
            .groupby("home_conference")["ats_correct"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "ATS Win %",
                    "count": "Bets",
                    "home_conference": "Conference",
                }
            )
            .sort_values("ATS Win %", ascending=False)
        )
        ats_by_conf = ats_by_conf.assign(
            **{"ATS Win %": lambda d: (d["ATS Win %"] * 100).round(1)}
        )
        ats_table = {
            "heading": "Model ATS Win % by Conference",
            "columns": [str(column) for column in ats_by_conf.columns],
            "rows": records(ats_by_conf),
        }

    return {
        "heading": "Season Trends",
        "score_figure": figure_json(fig_score),
        "hfa_figure": figure_json(fig_hfa),
        "ats_table": ats_table,
    }


def _h2h(df: pd.DataFrame, team_a: str, team_b: str) -> dict:
    heading = "Head-to-Head Lookup"
    mask = ((df["home_team"] == team_a) & (df["away_team"] == team_b)) | (
        (df["home_team"] == team_b) & (df["away_team"] == team_a)
    )
    h2h = df[mask].sort_values("season", ascending=False)

    if h2h.empty:
        return {
            "heading": heading,
            "info": f"No matchups found between {team_a} and {team_b} in the dataset.",
        }

    a_wins = int(
        ((h2h["home_team"] == team_a) & (h2h["home_margin"] > 0)).sum()
        + ((h2h["away_team"] == team_a) & (h2h["home_margin"] < 0)).sum()
    )
    b_wins = len(h2h) - a_wins

    rows = []
    for _, g in h2h.iterrows():
        is_a_home = g["home_team"] == team_a
        a_score = (
            int(g["home_score"] if is_a_home else g["away_score"])
            if pd.notna(g.get("home_score")) else None
        )
        b_score = (
            int(g["away_score"] if is_a_home else g["home_score"])
            if pd.notna(g.get("away_score")) else None
        )
        result = "—"
        if a_score is not None:
            result = f"{team_a} {a_score}–{b_score} {team_b}"
        rows.append(
            {
                "Season": int(g["season"]),
                "Week": int(g.get("week", 0)),
                "Site": "Neutral" if g.get("neutral_site") else ("Home" if is_a_home else "Away"),
                "Result": result,
                "Spread": (
                    f"{g.get('market_spread', float('nan')):+.1f}"
                    if pd.notna(g.get("market_spread")) else "—"
                ),
            }
        )

    return {
        "heading": heading,
        "metrics": [
            {"label": f"{team_a} wins", "value": str(a_wins), "delta": None, "help": None},
            {"label": f"{team_b} wins", "value": str(b_wins), "delta": None, "help": None},
            {"label": "Games", "value": str(len(h2h)), "delta": None, "help": None},
        ],
        "table": {
            "columns": ["Season", "Week", "Site", "Result", "Spread"],
            "rows": records(pd.DataFrame(rows)),
        },
    }


def _conference_power(df_slice: pd.DataFrame) -> dict:
    heading = "Conference Power"
    df_nc = df_slice[df_slice["conference_game"] == 0].copy()
    if df_nc.empty:
        return {
            "heading": heading,
            "info": "No non-conference game data available in the selected range.",
        }

    df_nc["home_conf"] = df_nc["home_conference"].fillna("Unknown")
    df_nc["away_conf"] = df_nc["away_conference"].fillna("Unknown")

    pivot_rows = []
    for conf, grp in df_nc.groupby("home_conf"):
        wins = (grp["home_margin"] > 0).sum()
        total = len(grp)
        pivot_rows.append(
            {
                "Conference": conf,
                "NC Win %": round(wins / total * 100, 1) if total else 0,
                "Games": total,
            }
        )

    conf_df = pd.DataFrame(pivot_rows).sort_values("NC Win %", ascending=False)
    fig_bar = px.bar(
        conf_df, x="Conference", y="NC Win %",
        title="Non-Conference Win % by Conference (home games)",
        color="NC Win %", color_continuous_scale="RdYlGn",
        range_color=[30, 70],
    )
    fig_bar.update_layout(
        paper_bgcolor="#F7FBFF", plot_bgcolor="#F7FBFF", font=dict(color="#1A2B3C")
    )

    conf_scoring = (
        df_slice.groupby("home_conference")[["home_score", "away_score"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "home_conference": "Conference",
                "home_score": "Avg Home Score",
                "away_score": "Avg Away Score",
            }
        )
        .dropna()
        .sort_values("Avg Home Score", ascending=False)
    )

    return {
        "heading": heading,
        "figure": figure_json(fig_bar),
        "table": {
            "heading": "Average Scoring by Conference",
            "columns": [str(column) for column in conf_scoring.round(1).columns],
            "rows": records(conf_scoring.round(1)),
        },
    }


def build_historical(
    season_from: int | None = None,
    season_to: int | None = None,
    team_a: str | None = None,
    team_b: str | None = None,
) -> dict:
    """Return the Historical Analysis payload."""
    df = load_data()

    base = {"page": "historical", "title": "📈 Historical Analysis"}
    if df.empty:
        return {**base, "warnings": ["No historical data is currently published."], "stopped": True}

    seasons = sorted(df["season"].dropna().unique())
    season_from = int(seasons[0]) if season_from is None else int(season_from)
    season_to = int(seasons[-1]) if season_to is None else int(season_to)
    if season_from not in seasons:
        season_from = int(seasons[0])
    if season_to not in seasons:
        season_to = int(seasons[-1])

    df_slice = df[(df["season"] >= season_from) & (df["season"] <= season_to)].copy()

    all_teams = sorted(
        set(df["home_team"].dropna().tolist() + df["away_team"].dropna().tolist())
    )
    if team_a is None or team_a not in all_teams:
        team_a = all_teams[all_teams.index("Ohio State")] if "Ohio State" in all_teams else all_teams[0]
    if team_b is None or team_b not in all_teams:
        team_b = all_teams[all_teams.index("Michigan")] if "Michigan" in all_teams else all_teams[1]

    return {
        **base,
        "seasons": [int(value) for value in seasons],
        "teams": all_teams,
        "selection": {
            "season_from": season_from,
            "season_to": season_to,
            "team_a": team_a,
            "team_b": team_b,
        },
        "tabs": [
            {"id": "trends", "label": "📅 Season Trends", **_season_trends(df_slice)},
            {"id": "h2h", "label": "🆚 H2H Lookup", **_h2h(df, team_a, team_b)},
            {"id": "conf", "label": "🏆 Conference Power", **_conference_power(df_slice)},
        ],
        "footer": True,
    }
