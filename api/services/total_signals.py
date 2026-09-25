"""Total Market Signals — mirrors ``pages/10_Total_Market_Signals.py``."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from api.data import DATA_DIR, parquet
from api.jsonutil import records
from utils.model_artifacts import load_metrics

SHADOW_PATH = DATA_DIR / "shadow_total_signals.json"


def build_total_signals(season: int | None = None) -> dict:
    """Return the Total Market Signals payload."""
    base = {
        "page": "total_signals",
        "title": "Total Market Signals",
        "caption": (
            "A separate closing-time over/under classifier using opening-to-current movement, "
            "cross-book dispersion, market depth, and point-in-time team form."
        ),
    }

    metrics = load_metrics().get("total_cover_model", {})
    strategy = metrics.get("strategy", {})
    release = metrics.get("strategy_release", {})
    eligible_sides = release.get("eligible_sides", [])

    if not metrics:
        return {**base, "warnings": ["The total-side artifact has not been trained."], "stopped": True}

    status = str(release.get("status", "hold")).upper()
    warnings: list[str] = []
    errors: list[str] = []
    if status == "SHADOW":
        warnings.append(
            "Shadow deployment only. Retrospective gates passed, but no signal becomes "
            "an automated bet until 2026 prospective closing-line value passes."
        )
    else:
        errors.append("The total-side strategy is on hold.")

    over_metrics = strategy.get("by_side", {}).get("over", {})
    confirmation = strategy.get("by_season_and_side", {}).get("2025", {}).get("over", {})

    metric_cards = [
        {
            "label": "OOS Brier",
            "value": f"{metrics.get('brier', float('nan')):.4f}",
            "delta": "vs 0.2500",
            "help": None,
        },
        {
            "label": "Validated Over Record",
            "value": f"{over_metrics.get('wins', 0)}–{over_metrics.get('losses', 0)}",
            "delta": f"{over_metrics.get('win_rate', 0):.1%}",
            "help": None,
        },
        {
            "label": "2025 Confirmation",
            "value": f"{confirmation.get('wins', 0)}–{confirmation.get('losses', 0)}",
            "delta": f"{confirmation.get('win_rate', 0):.1%}",
            "help": None,
        },
        {
            "label": "Shadow-Eligible Side",
            "value": ", ".join(side.title() for side in eligible_sides) or "None",
            "delta": None,
            "help": None,
        },
    ]

    strategy_caption = (
        f"Selection requires probability ≥ {strategy.get('minimum_selected_probability', 0.575):.1%}. "
        "Historical ROI assumes every price was -110; actual deployment requires captured executable prices."
    )

    try:
        backtest = parquet("model_backtest", layer="features")
    except FileNotFoundError:
        backtest = pd.DataFrame()

    decisions: dict = {"heading": "Walk-forward decisions"}
    if backtest.empty or "total_over_prob_oos" not in backtest:
        decisions["info"] = "No total-side OOS artifact is available."
    else:
        threshold = float(strategy.get("probability_edge_threshold", 0.075))
        selected = backtest[
            backtest["total_over_prob_oos"].notna()
            & ((backtest["total_over_prob_oos"] - 0.5).abs() >= threshold)
        ].copy()
        selected["Side"] = np.where(selected["total_over_prob_oos"] > 0.5, "Over", "Under")
        selected = selected[selected["Side"].str.lower().isin(eligible_sides)]

        seasons = sorted(selected["season"].dropna().unique(), reverse=True)
        # The Streamlit page only renders the season selectbox when a held-out
        # season exists, and only then narrows the frame.
        season = (
            int(season) if season is not None and season in seasons
            else (int(seasons[0]) if seasons else None)
        )
        if season is not None:
            selected = selected[selected["season"] == season].copy()

        selected["Game"] = (
            selected.get("away_team", selected["game_id"].astype(str)).astype(str)
            + " @ "
            + selected.get("home_team", "").astype(str)
        )
        selected["Probability"] = np.where(
            selected["Side"].eq("Over"),
            selected["total_over_prob_oos"],
            1 - selected["total_over_prob_oos"],
        )
        selected["Result"] = np.where(
            np.where(
                selected["Side"].eq("Over"),
                selected["total_points"] > selected["market_total"],
                selected["total_points"] < selected["market_total"],
            ),
            "WIN",
            "LOSS",
        )
        columns = [
            column for column in (
                "week", "Game", "Side", "market_total", "market_total_open",
                "market_total_move", "Probability", "Result",
            ) if column in selected
        ]
        table = selected[columns].sort_values("Probability", ascending=False)
        decisions.update(
            {
                "seasons": [int(value) for value in seasons],
                "season": season if seasons else None,
                "columns": columns,
                "rows": records(table),
            }
        )

    shadow = (
        json.loads(SHADOW_PATH.read_text(encoding="utf-8"))
        if SHADOW_PATH.exists() else {}
    )
    signals = pd.DataFrame(shadow.get("signals", []))
    shadow_block: dict = {"heading": "Current prospective shadow file"}
    if signals.empty:
        shadow_block["info"] = shadow.get("meta", {}).get(
            "note", "No closing-time shadow signals currently qualify."
        )
    else:
        shadow_block["table"] = {
            "columns": [str(column) for column in signals.columns],
            "rows": records(signals),
        }

    return {
        **base,
        "warnings": warnings,
        "errors": errors,
        "metrics": metric_cards,
        "strategy_caption": strategy_caption,
        "decisions": decisions,
        "shadow": shadow_block,
        "footer": True,
    }
