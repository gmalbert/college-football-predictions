"""Model Performance — mirrors ``pages/5_Model_Performance.py``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from api.charts import figure_json
from api.data import parquet
from api.jsonutil import jsonable, records
from utils.feature_engine import SPREAD_FEATURES, TOTAL_FEATURES
from utils.models import WIN_MODEL_PATH, load_metrics, load_models, models_trained

try:
    from sklearn.calibration import calibration_curve

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


def _metric_better(value, baseline) -> bool:
    try:
        return (
            np.isfinite(float(value))
            and np.isfinite(float(baseline))
            and float(value) < float(baseline)
        )
    except (TypeError, ValueError):
        return False


def load_backtest() -> pd.DataFrame:
    try:
        return parquet("model_backtest", layer="features")
    except FileNotFoundError:
        return pd.DataFrame()


def _calibration_figure(backtest: pd.DataFrame) -> dict | None:
    df_cal = backtest.dropna(subset=["win_prob_oos", "home_win"]).copy()
    if df_cal.empty:
        return None
    probs = df_cal["win_prob_oos"].to_numpy()
    y_true = df_cal["home_win"].values.astype(int)
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mean_pred, y=frac_pos,
            mode="lines+markers",
            line=dict(color="#D4001C", width=2),
            marker=dict(size=8),
            name="Model",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Perfect calibration",
        )
    )
    fig.update_layout(
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        paper_bgcolor="#F7FBFF",
        plot_bgcolor="#F7FBFF",
        font=dict(color="#1A2B3C"),
    )
    return figure_json(fig)


def _ats_week_figure(backtest: pd.DataFrame) -> dict | None:
    if backtest.empty or "predicted_spread_oos" not in backtest.columns:
        return None
    sub = backtest.dropna(
        subset=["predicted_spread_oos", "market_spread", "home_margin"]
    ).copy()
    sub = sub[(sub["home_margin"] + sub["market_spread"]) != 0]
    if sub.empty:
        return None
    sub["model_picks_home"] = sub["predicted_spread_oos"] > -sub["market_spread"]
    sub["home_covered"] = sub["home_margin"] > -sub["market_spread"]
    sub["correct"] = sub["model_picks_home"] == sub["home_covered"]

    ats_week = (
        sub.groupby("week")["correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "win_pct", "count": "n", "week": "Week"})
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ats_week["Week"], y=ats_week["win_pct"] * 100,
            marker_color=["#D4001C" if v >= 52.4 else "#333" for v in ats_week["win_pct"] * 100],
            name="ATS Win %",
        )
    )
    fig.add_hline(
        y=52.4, line_dash="dash", line_color="gold",
        annotation_text="Break-even (52.4%)",
    )
    fig.update_layout(
        yaxis_title="ATS Win %",
        xaxis_title="Week",
        paper_bgcolor="#F7FBFF",
        plot_bgcolor="#F7FBFF",
        font=dict(color="#1A2B3C"),
    )
    return figure_json(fig)


def _importance(model, features: list[str]) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame()
    if HAS_XGB and isinstance(model, xgb.Booster):
        raw_imp = model.get_score(importance_type="gain")
        return pd.DataFrame(
            [
                {
                    "Feature": feature,
                    "Importance": raw_imp.get(feature, raw_imp.get(f"f{i}", 0)),
                }
                for i, feature in enumerate(features)
            ]
        ).sort_values("Importance", ascending=True)
    try:
        coefs = abs(model.named_steps["ridge"].coef_)
        used_feats = model.named_steps["imputer"].get_feature_names_out(features)
        return pd.DataFrame(
            {"Feature": used_feats, "Importance": coefs}
        ).sort_values("Importance")
    except Exception:  # noqa: BLE001 - parity with the Streamlit fallback
        return pd.DataFrame()


def _importance_figure(imp_df: pd.DataFrame) -> dict | None:
    if imp_df.empty:
        return None
    fig = go.Figure(
        go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"],
            orientation="h", marker_color="#D4001C",
        )
    )
    fig.update_layout(
        xaxis_title="Importance",
        paper_bgcolor="#F7FBFF",
        plot_bgcolor="#F7FBFF",
        font=dict(color="#1A2B3C"),
    )
    return figure_json(fig)


SPREAD_CATALOG = [
    ("elo_diff", "Home team Elo rating minus away team Elo rating.", "Captures overall team strength and matchup quality."),
    ("sp_plus_diff", "SP+ overall rating difference (home minus away).", "Adds a modern analytics rating that blends offense, defense, and schedule."),
    ("sp_offense_diff", "SP+ offensive efficiency difference.", "Highlights which team has the more productive offense."),
    ("sp_defense_diff", "SP+ defensive efficiency difference.", "Reflects defensive capability to limit opponents."),
    ("off_epa_diff", "Offensive EPA per play difference.", "Measures efficiency by scoring value per play."),
    ("def_epa_diff", "Defensive EPA per play difference.", "Captures how well a defense prevents high-value plays."),
    ("off_explosiveness_diff", "Explosiveness metric difference.", "Shows the team likely to generate big plays."),
    ("def_havoc_diff", "Defensive havoc metric difference.", "Indicates turnover pressure and disruption ability."),
    ("off_success_diff", "Offensive success rate difference.", "Represents consistency in gaining needed yardage."),
    ("off_rushing_epa_diff", "Rushing EPA per play difference.", "Reveals the strength of the run game advantage."),
    ("off_passing_epa_diff", "Passing EPA per play difference.", "Reveals the strength of the passing game advantage."),
    ("recruiting_diff", "Recruiting score/talent difference.", "Proxy for roster talent depth and future upside."),
    ("talent_diff", "Overall talent rating difference.", "General strength gap between home and away rosters."),
    ("recruiting_rank_diff", "Recruiting class rank difference.", "Indicates relative talent quality by incoming classes."),
    ("home_flag", "Indicator for the home team.", "Captures home-field advantage effects."),
    ("conference_game", "Indicator for conference matchup.", "Accounts for rivalry / familiarity effects."),
    ("rest_advantage", "Home rest days minus away rest days.", "Models fatigue or recovery advantage."),
    ("turnover_margin_l5", "Last 5 games turnover margin difference.", "Reflects recent ability to create and avoid turnovers."),
    ("rushing_yards_diff_l5", "Last 5 games rushing yards difference.", "Tracks recent ground-game dominance."),
    ("pass_yards_diff_l5", "Last 5 games passing yards difference.", "Tracks recent aerial-attack dominance."),
    ("penalty_yards_diff_l5", "Last 5 games penalty yards difference.", "Captures discipline and self-inflicted disadvantage."),
    ("fpi_diff", "Home minus away FPI rating difference.", "Adds a consensus national power-rating signal."),
    ("srs_diff", "Home minus away SRS rating difference.", "Adds a margin-adjusted, SOS-aware rating signal."),
    ("returning_ppa_diff", "Returning production percentage difference.", "Captures roster continuity and experience."),
    ("ppa_off_diff", "Offensive PPA difference between teams.", "Reflects opponent-adjusted offensive efficiency."),
    ("ppa_def_diff", "Defensive PPA difference between teams.", "Reflects opponent-adjusted defensive efficiency."),
    ("ppa_third_down_off_diff", "Third-down offensive PPA difference.", "Measures efficiency in critical third-down situations."),
    ("ppa_third_down_def_diff", "Third-down defensive PPA difference.", "Measures ability to stop opponents on third downs."),
    ("wepa_off_diff", "Opponent-adjusted offensive EPA difference.", "Accounts for schedule strength and tempo-adjusted offense."),
    ("wepa_def_diff", "Opponent-adjusted defensive EPA difference.", "Accounts for schedule strength and tempo-adjusted defense."),
    ("cfbd_pregame_wp_diff", "CFBD pre-game home win probability minus 0.5.", "Provides a consensus probability baseline for the matchup."),
    ("coach_tenure_diff", "Home coach tenure minus away coach tenure.", "Models coaching experience and first-year coach risk."),
    ("market_spread", "Closing market spread, used as the consensus line.", "Anchors the model to the betting market and line value."),
]

TOTAL_CATALOG = [
    ("home_off_epa", "Home offense EPA per play.", "Measures home team's scoring efficiency."),
    ("away_off_epa", "Away offense EPA per play.", "Measures away team's scoring efficiency."),
    ("home_def_epa", "Home defense EPA allowed per play.", "Indicates how well the home defense limits opponents."),
    ("away_def_epa", "Away defense EPA allowed per play.", "Indicates how well the away defense limits opponents."),
    ("home_off_explosiveness", "Home offensive explosiveness.", "Captures big-play scoring upside at home."),
    ("away_off_explosiveness", "Away offensive explosiveness.", "Captures big-play scoring upside on the road."),
    ("home_off_rushing_epa", "Home rushing EPA per play.", "Measures home running game scoring value."),
    ("away_off_rushing_epa", "Away rushing EPA per play.", "Measures away running game scoring value."),
    ("home_off_passing_epa", "Home passing EPA per play.", "Measures home passing game scoring value."),
    ("away_off_passing_epa", "Away passing EPA per play.", "Measures away passing game scoring value."),
    ("home_flag", "Indicator for the home team.", "Captures home-field scoring advantage."),
    ("rest_days_home", "Days of rest for home team.", "Models freshness and recovery effects."),
    ("rest_days_away", "Days of rest for away team.", "Models fatigue and travel effects."),
    ("market_total", "Betting market total line.", "Anchors the model to market scoring expectations."),
    ("is_dome", "Indoor/dome game flag.", "Adjusts for weather-insulated scoring environments."),
    ("temperature", "Forecast temperature.", "Cold games usually suppress scoring."),
    ("wind_speed", "Forecast wind speed.", "High wind reduces passing efficiency and scoring."),
    ("adverse_weather", "Bad weather indicator.", "Captures rain, snow, wind, or cold impact on scoring."),
    ("high_wind", "High-wind indicator.", "Highlights games where kicking and passing suffer."),
    ("high_altitude", "High elevation venue flag.", "Models altitude effects on scoring and endurance."),
    ("is_primetime", "Prime-time game flag.", "Captures TV/prime-time scoring and officiating effects."),
]


def build_model_performance() -> dict:
    """Return the Model Performance payload."""
    base = {"page": "model_performance", "title": "🎯 Model Performance"}

    if not models_trained():
        return {
            **base,
            "warnings": ["Model evaluation artifacts are not currently published."],
            "stopped": True,
        }

    metrics = load_metrics()
    errors: list[str] = []
    if metrics.get("evaluation_scope") != "walk_forward_season_oos":
        errors.append(
            "Saved metrics predate the leakage-safe walk-forward evaluator. "
            "Retrain before using this page for betting decisions."
        )

    win_m = metrics.get("win_model", {})
    spread_m = metrics.get("spread_model", {})
    total_m = metrics.get("total_model", {})
    ats_m = metrics.get("ats", {})

    comparison = pd.DataFrame(
        [
            {
                "Target": "Win probability (Brier)",
                "Comparable model": win_m.get("model_brier_on_baseline_subset"),
                "Market baseline": win_m.get("baseline_brier"),
                "Priced games": win_m.get("baseline_n"),
                "Model beats market": _metric_better(
                    win_m.get("model_brier_on_baseline_subset"), win_m.get("baseline_brier")
                ),
            },
            {
                "Target": "Home margin (RMSE)",
                "Comparable model": spread_m.get("model_rmse_on_baseline_subset"),
                "Market baseline": spread_m.get("baseline_rmse"),
                "Priced games": spread_m.get("baseline_n"),
                "Model beats market": _metric_better(
                    spread_m.get("model_rmse_on_baseline_subset"), spread_m.get("baseline_rmse")
                ),
            },
            {
                "Target": "Game total (RMSE)",
                "Comparable model": total_m.get("model_rmse_on_baseline_subset"),
                "Market baseline": total_m.get("baseline_rmse"),
                "Priced games": total_m.get("baseline_n"),
                "Model beats market": _metric_better(
                    total_m.get("model_rmse_on_baseline_subset"), total_m.get("baseline_rmse")
                ),
            },
        ]
    )

    gates = pd.DataFrame(metrics.get("release_gates", []))
    if not gates.empty:
        gates["threshold"] = gates["threshold"].map(str)

    backtest = load_backtest()
    models = load_models()

    calibration = None
    if not backtest.empty and HAS_SKLEARN:
        calibration = _calibration_figure(backtest)

    ats_week = _ats_week_figure(backtest)

    spread_imp = _importance(models.get("spread"), SPREAD_FEATURES)
    total_imp = _importance(models.get("total"), TOTAL_FEATURES)

    return {
        **base,
        "errors": errors,
        "metrics": [
            {"label": "OOS Brier Score", "value": f"{win_m.get('brier', 0):.4f}", "delta": None, "help": "Walk-forward; lower is better"},
            {"label": "OOS Spread RMSE", "value": f"{spread_m.get('rmse', 0):.2f} pts", "delta": None, "help": "Walk-forward by season"},
            {"label": "OOS Total RMSE", "value": f"{total_m.get('rmse', 0):.2f} pts", "delta": None, "help": "Walk-forward by season"},
            {"label": "OOS ATS Win %", "value": f"{ats_m.get('pct', 0):.1%}", "delta": None, "help": "Break-even is 52.4%"},
            {"label": "ATS Record", "value": f"{ats_m.get('wins', 0)}‑{ats_m.get('losses', 0)}", "delta": None, "help": None},
        ],
        "comparison_caption": "Market comparisons use exactly the same priced games for model and baseline.",
        "comparison": {
            "columns": [str(column) for column in comparison.columns],
            "rows": records(comparison),
        },
        "gates_expander": {
            "label": "Release gates",
            "expanded": False,
            "table": None if gates.empty else {
                "columns": [str(column) for column in gates.columns],
                "rows": records(gates),
            },
            "info": "No release-gate results are saved." if gates.empty else None,
        },
        "calibration": {
            "heading": "Calibration Curve — Win Probability",
            "figure": calibration,
            "info": None if calibration else (
                "No out-of-sample win predictions are available."
                if not backtest.empty else None
            ),
        },
        "ats_week": {
            "heading": "ATS Record by Week",
            "figure": ats_week,
        },
        "spread_importance": {
            "heading": "Feature Importance — Spread Model",
            "figure": _importance_figure(spread_imp),
            "info": (
                "Feature importance not available for this model type."
                if spread_imp.empty else None
            ),
            "catalog": {
                "label": "Spread Feature Catalog",
                "info": (
                    "Active production inputs: " + ", ".join(SPREAD_FEATURES) +
                    ". The table below is an extended source catalog; season-final fields are "
                    "excluded from production unless they gain a point-in-time available_at timestamp."
                ),
                "rows": [list(row) for row in SPREAD_CATALOG],
            },
        },
        "total_importance": {
            "heading": "Feature Importance — Total Model",
            "figure": _importance_figure(total_imp),
            "info": (
                "Feature importance not available for this model type."
                if total_imp.empty else None
            ),
            "catalog": {
                "label": "Total Feature Catalog",
                "info": (
                    "Active production inputs: " + ", ".join(TOTAL_FEATURES) +
                    ". Weather and venue candidates below activate only when an auditable pregame snapshot exists."
                ),
                "rows": [list(row) for row in TOTAL_CATALOG],
            },
        },
        "summary_heading": "Training Data Summary",
        "summary_metrics": [
            {"label": "Win model samples", "value": f"{win_m.get('n_samples'):,}" if win_m.get("n_samples") else "—", "delta": None, "help": None},
            {"label": "Spread model samples", "value": f"{spread_m.get('n_samples'):,}" if spread_m.get("n_samples") else "—", "delta": None, "help": None},
            {"label": "Total model samples", "value": f"{total_m.get('n_samples'):,}" if total_m.get("n_samples") else "—", "delta": None, "help": None},
        ],
        "footer": True,
        "meta": jsonable({"win_model_path": WIN_MODEL_PATH.name}),
    }
