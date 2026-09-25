"""Win Probability — mirrors ``pages/7_Win_Probability.py``."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from api.charts import figure_json
from api.data import memo
from api.jsonutil import records
from utils.cfbd_client import (
    get_games,
    get_win_probability_chart,
    parse_win_probability_rows,
)

SEASON_OPTIONS = list(range(2025, 2020, -1))
SEASON_TYPES = ["Regular", "Postseason"]


def _week_options(season_type: str) -> list[int]:
    return list(range(1, 18)) if season_type == "regular" else list(range(1, 6))


def _load_games(year: int, week: int, season_type: str) -> list:
    return memo(
        f"cfbd:games:{year}:{season_type}:{week}",
        lambda: get_games(year, season_type=season_type, week=week),
    )


def _load_wp(game_id: int) -> list:
    return memo(
        f"cfbd:wp:{game_id}",
        lambda: get_win_probability_chart(game_id),
    )


def _game_label(game) -> tuple[str | None, str, str]:
    def _get(name: str, default):
        if isinstance(game, dict):
            return game.get(name, default)
        return getattr(game, name, default)

    home = _get("home_team", "?")
    away = _get("away_team", "?")
    gid = _get("id", None)
    return gid, str(away), str(home)


def _wp_figure(df: pd.DataFrame, home_name: str, away_name: str) -> dict:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["play"], y=df["home_wp"],
            fill="tozeroy", fillcolor="rgba(30, 100, 200, 0.15)",
            line=dict(color="rgba(30, 100, 200, 0.8)", width=2),
            name=f"{home_name} WP",
            hovertemplate=(
                "Play %{x}<br>"
                f"{home_name} Win Prob: %{{y:.1%}}<br>"
                "%{customdata}<extra></extra>"
            ),
            customdata=df["play_text"],
        )
    )
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="gray",
        annotation_text="50%", annotation_position="right",
    )
    total_plays = len(df)
    for q, label in [
        (total_plays // 4, "Q2"),
        (total_plays // 2, "Q3"),
        (3 * total_plays // 4, "Q4"),
    ]:
        fig.add_vline(
            x=df["play"].iloc[min(q, len(df) - 1)],
            line_dash="dot", line_color="lightgray", opacity=0.6,
        )
        fig.add_annotation(
            x=df["play"].iloc[min(q, len(df) - 1)],
            y=0.95, text=label, showarrow=False,
            font=dict(size=10, color="gray"),
        )
    fig.update_layout(
        title=f"{away_name} @ {home_name} — Win Probability",
        xaxis_title="Play Number",
        yaxis_title=f"{home_name} Win Probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=480,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="#F7FBFF",
        plot_bgcolor="#F7FBFF",
    )
    return figure_json(fig)


def build_win_probability(
    season: int = 2025,
    season_type_display: str = "Regular",
    week: int = 1,
    search: str = "",
    game_id: int | None = None,
) -> dict:
    """Return the Win Probability payload."""
    base = {
        "page": "win_probability",
        "title": "📈 In-Game Win Probability",
        "caption": (
            "Play-by-play home-team win probability from the CFBD model. "
            "Select a season and week, then pick a game."
        ),
    }

    if season not in SEASON_OPTIONS:
        season = SEASON_OPTIONS[0]
    if season_type_display not in SEASON_TYPES:
        season_type_display = SEASON_TYPES[0]
    season_type = season_type_display.lower()

    week_options = _week_options(season_type)
    if week not in week_options:
        week = week_options[0]

    controls = {
        "seasons": SEASON_OPTIONS,
        "season": season,
        "season_types": SEASON_TYPES,
        "season_type": season_type_display,
        "weeks": week_options,
        "week": week,
        "search": search,
    }

    try:
        games = _load_games(season, week, season_type)
    except Exception as exc:  # noqa: BLE001 - mirrors the Streamlit warning
        return {
            **base,
            "controls": controls,
            "warnings": [
                "Could not load games — check that the CFBD API key is configured "
                f"in Streamlit secrets.  \n`{exc}`"
            ],
            "stopped": True,
        }

    if not games:
        return {**base, "controls": controls, "info": "No games found for the selected week.", "stopped": True}

    game_options: dict[str, int] = {}
    for game in games:
        gid, away, home = _game_label(game)
        if gid:
            game_options[f"{away} @ {home}"] = gid

    if not game_options:
        return {**base, "controls": controls, "info": "No games with IDs found.", "stopped": True}

    filtered = sorted(
        key for key in game_options if not search or search.lower() in key.lower()
    )
    if not filtered:
        return {
            **base,
            "controls": controls,
            "warnings": [f"No games matching '{search}'."],
            "stopped": True,
        }

    selected_label = (
        next((label for label, value in game_options.items() if value == game_id), None)
        if game_id is not None else None
    )
    if selected_label is None or selected_label not in filtered:
        selected_label = filtered[0]
    resolved_game_id = game_options[selected_label]

    controls.update(
        {
            "games": filtered,
            "game_options": game_options,
            "selected_label": selected_label,
            "game_id": resolved_game_id,
            "game_id_caption": f"Game ID: `{resolved_game_id}`",
        }
    )

    try:
        wp_data = _load_wp(resolved_game_id)
    except Exception:  # noqa: BLE001
        wp_data = []

    if not wp_data:
        return {
            **base,
            "controls": controls,
            "warnings": [
                "No win probability data available for this game. "
                "CFBD only provides this for completed games."
            ],
            "stopped": True,
        }

    parsed = parse_win_probability_rows(wp_data)
    if not parsed:
        return {
            **base,
            "controls": controls,
            "warnings": ["Win probability data could not be parsed."],
            "stopped": True,
        }

    df = pd.DataFrame(parsed).sort_values("play")

    parts = selected_label.split(" @ ")
    away_name = parts[0] if len(parts) == 2 else "Away"
    home_name = parts[1] if len(parts) == 2 else "Home"

    final_wp = float(df["home_wp"].iloc[-1])
    metrics = [
        {"label": f"{home_name} Win Prob", "value": f"{final_wp:.1%}", "delta": None, "help": None},
        {"label": f"{away_name} Win Prob", "value": f"{1 - final_wp:.1%}", "delta": None, "help": None},
    ]
    if df["home_score"].notna().any():
        metrics.append(
            {
                "label": "Final Score",
                "value": (
                    f"{home_name} {int(df['home_score'].iloc[-1] or 0)}  –  "
                    f"{int(df['away_score'].iloc[-1] or 0)}  {away_name}"
                ),
                "delta": None,
                "help": None,
            }
        )

    return {
        **base,
        "controls": controls,
        "metrics": metrics,
        "figure": _wp_figure(df, home_name, away_name),
        "raw_expander": {
            "label": "Raw play data",
            "expanded": False,
            "columns": ["play", "home_wp", "home_score", "away_score", "play_text"],
            "rows": records(df),
            "height": 300,
        },
    }
