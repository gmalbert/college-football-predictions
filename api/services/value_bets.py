"""Value Bets — mirrors ``pages/2_Value_Bets.py``."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from api.charts import figure_json
from api.data import parquet
from api.jsonutil import records
from utils.betting import (
    CONFIDENCE_LABEL,
    Confidence,
    generate_moneyline_pick,
    generate_spread_pick,
    generate_total_pick,
    half_kelly,
    kelly_fraction,
    simulate_bankroll,
)
from utils.models import load_metrics, models_trained, predict_for_display


def load_data() -> pd.DataFrame:
    try:
        return parquet("feature_matrix", layer="features")
    except FileNotFoundError:
        return pd.DataFrame()


def _conf_gte(c: Confidence, minimum: Confidence) -> bool:
    order = [Confidence.LEAN, Confidence.MODERATE, Confidence.STRONG]
    return order.index(c) >= order.index(minimum)


def _spread_result(margin: float, book: float, pick_home: bool) -> str:
    settled = margin + book
    if not pick_home:
        settled = -settled
    if settled > 0:
        return "✅ WIN"
    if settled == 0:
        return "➡️ PUSH"
    return "❌ LOSS"


def _total_result(total: float, book: float, model: float) -> str:
    over = model > book
    if (over and total > book) or (not over and total < book):
        return "✅ WIN"
    if total == book:
        return "➡️ PUSH"
    return "❌ LOSS"


def _ml_result(home: str, away: str, margin: float, pick: str) -> str:
    if pd.isna(margin):
        return "—"
    home_won = margin > 0
    away_won = margin < 0
    if home in pick:
        return "✅ WIN" if home_won else ("➡️ PUSH" if margin == 0 else "❌ LOSS")
    if away in pick:
        return "✅ WIN" if away_won else ("➡️ PUSH" if margin == 0 else "❌ LOSS")
    return "—"


CONF_ORDER = {
    "STRONG": Confidence.STRONG,
    "MODERATE": Confidence.MODERATE,
    "LEAN": Confidence.LEAN,
    "ALL": None,
}


def _bankroll_chart(
    start_bankroll: float,
    stake_method: str,
    bet_odds: float,
    scenario_probability: float,
    df_recs: pd.DataFrame,
) -> dict | None:
    sim_bets = []
    for _, row in df_recs[df_recs["Result"].isin(["✅ WIN", "❌ LOSS"])].iterrows():
        wp = scenario_probability
        if stake_method == "Half Kelly":
            stake_frac = half_kelly(wp, bet_odds)
        elif stake_method == "Full Kelly":
            stake_frac = kelly_fraction(wp, bet_odds)
        else:
            stake_frac = 0.01
        sim_bets.append(
            {
                "result": "W" if row["Result"] == "✅ WIN" else "L",
                "odds": bet_odds,
                "stake": stake_frac,
            }
        )

    if not sim_bets:
        return None

    bankroll_curve = simulate_bankroll(sim_bets, starting=float(start_bankroll))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=bankroll_curve,
            mode="lines",
            line=dict(color="#D4001C", width=2),
            fill="tozeroy",
            fillcolor="rgba(212,0,28,0.1)",
            name="Bankroll",
        )
    )
    fig.add_hline(
        y=start_bankroll, line_dash="dash", line_color="gray",
        annotation_text="Starting bankroll",
    )
    fig.update_layout(
        title="Cumulative Bankroll",
        xaxis_title="Bet #",
        yaxis_title="Bankroll ($)",
        paper_bgcolor="#F7FBFF",
        plot_bgcolor="#F7FBFF",
        font=dict(color="#1A2B3C"),
    )
    final = bankroll_curve[-1]
    roi = (final - start_bankroll) / start_bankroll
    return {
        "figure": figure_json(fig),
        "metrics": [
            {"label": "Final Bankroll", "value": f"${final:,.0f}", "delta": f"{roi:+.1%}", "help": None},
            {"label": "Net P&L", "value": f"${final - start_bankroll:+,.0f}", "delta": None, "help": None},
        ],
    }


def build_value_bets(
    season: int | None = None,
    bet_type: str = "Spread",
    min_edge: float = 2.0,
    min_conf: str = "MODERATE",
    start_bankroll: float = 1000,
    stake_method: str = "Flat (1%)",
    bet_odds: float = -110,
    scenario_probability: float = 0.50,
) -> dict:
    """Return the Value Bets payload."""
    df_all = load_data()

    base = {
        "page": "value_bets",
        "title": "💰 Value Bets",
        "caption": "Games where the model's projected line disagrees materially with the sportsbook.",
    }

    if df_all.empty or not models_trained():
        return {
            **base,
            "warnings": ["No published data or model artifacts are currently available."],
            "stopped": True,
        }

    release = load_metrics().get("release_decision", {})
    warnings = []
    if release.get("decision") != "promote":
        failed = ", ".join(release.get("failed_gates", [])) or "release gates unavailable"
        warnings.append(
            f"Research mode only: the model release is on hold ({failed}). "
            "Do not interpret point differences as validated betting recommendations."
        )

    seasons = sorted(df_all["season"].dropna().unique(), reverse=True)
    season = int(seasons[0]) if season is None else int(season)
    if season not in seasons:
        season = int(seasons[0])

    df_season = df_all[df_all["season"] == season].copy()
    df_season = predict_for_display(df_season)

    captions: list[str] = []
    if "walk_forward_oos" in set(df_season["prediction_scope"].dropna()):
        captions.append(
            "Settled games use season walk-forward predictions; "
            "unplayed games use the current full-history model."
        )

    min_conf_enum = CONF_ORDER.get(min_conf)

    recs = []
    for _, row in df_season.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        gid = row.get("game_id")
        ms = row.get("predicted_spread")
        bs = row.get("market_spread")
        mt = row.get("predicted_total")
        bt = row.get("market_total")
        wp = row.get("win_prob", 0.5)
        actual_margin = row.get("home_margin")
        actual_total = row.get("total_points")

        if bet_type in ("Spread", "All") and pd.notna(ms) and pd.notna(bs):
            rec = generate_spread_pick(home, away, ms, bs, game_id=gid)
            if rec.edge >= min_edge and (
                min_conf_enum is None or _conf_gte(rec.confidence, min_conf_enum)
            ):
                recs.append(
                    {
                        "Week": int(row.get("week", 0)),
                        "Game": f"{away} @ {home}",
                        "Bet Type": "Spread",
                        "Model": f"{ms:+.1f}",
                        "Book": f"{bs:+.1f}",
                        "Edge": round(rec.edge, 1),
                        "Edge Tier": CONFIDENCE_LABEL[rec.confidence],
                        "Pick": rec.pick,
                        "Result": (
                            _spread_result(actual_margin, bs, rec.pick.startswith(f"Take {home} "))
                            if pd.notna(actual_margin) else "—"
                        ),
                        "_conf_val": rec.confidence,
                    }
                )

        if bet_type in ("Total", "All") and pd.notna(mt) and pd.notna(bt):
            rec = generate_total_pick(home, away, mt, bt, game_id=gid)
            if rec.edge >= min_edge and (
                min_conf_enum is None or _conf_gte(rec.confidence, min_conf_enum)
            ):
                recs.append(
                    {
                        "Week": int(row.get("week", 0)),
                        "Game": f"{away} @ {home}",
                        "Bet Type": "Total",
                        "Model": f"{mt:.1f}",
                        "Book": f"{bt:.1f}",
                        "Edge": round(rec.edge, 1),
                        "Edge Tier": CONFIDENCE_LABEL[rec.confidence],
                        "Pick": rec.pick,
                        "Result": (
                            _total_result(actual_total, bt, mt)
                            if pd.notna(actual_total) else "—"
                        ),
                        "_conf_val": rec.confidence,
                    }
                )

        if bet_type in ("Moneyline", "All") and pd.notna(wp):
            hml = row.get("home_moneyline")
            aml = row.get("away_moneyline")
            if pd.notna(hml) and pd.notna(aml):
                rec = generate_moneyline_pick(home, away, wp, float(hml), float(aml), game_id=gid)
                if rec is not None and (
                    min_conf_enum is None or _conf_gte(rec.confidence, min_conf_enum)
                ):
                    recs.append(
                        {
                            "Week": int(row.get("week", 0)),
                            "Game": f"{away} @ {home}",
                            "Bet Type": "Moneyline",
                            "Model": f"{wp:.1%}",
                            "Book": f"{int(hml):+d} / {int(aml):+d}",
                            "Edge": round(rec.edge * 100, 1),
                            "Edge Tier": CONFIDENCE_LABEL[rec.confidence],
                            "Pick": rec.pick,
                            "Result": (
                                _ml_result(home, away, actual_margin, rec.pick)
                                if pd.notna(actual_margin) else "—"
                            ),
                            "_conf_val": rec.confidence,
                        }
                    )

    controls = {
        "seasons": [int(value) for value in seasons],
        "season": season,
        "bet_types": ["Spread", "Total", "Moneyline", "All"],
        "bet_type": bet_type,
        "min_edge": min_edge,
        "min_conf_options": ["All", "LEAN", "MODERATE", "STRONG"],
        "min_conf": min_conf,
        "stake_methods": ["Flat (1%)", "Half Kelly", "Full Kelly"],
        "stake_method": stake_method,
        "start_bankroll": start_bankroll,
        "bet_odds": bet_odds,
        "scenario_probability": scenario_probability,
    }

    if not recs:
        return {
            **base,
            "warnings": warnings,
            "captions": captions,
            "controls": controls,
            "info": f"No bets found with edge ≥ {min_edge} pts and confidence ≥ {min_conf}.",
            "stopped": True,
        }

    df_recs = pd.DataFrame(recs).sort_values(["Edge"], ascending=False)

    wins = int((df_recs["Result"] == "✅ WIN").sum())
    losses = int((df_recs["Result"] == "❌ LOSS").sum())
    total = wins + losses

    display_cols = ["Week", "Game", "Bet Type", "Model", "Book", "Edge", "Edge Tier", "Pick", "Result"]
    table = df_recs[display_cols].reset_index(drop=True)

    chart = _bankroll_chart(
        start_bankroll, stake_method, bet_odds, scenario_probability, df_recs
    )

    return {
        **base,
        "warnings": warnings,
        "captions": captions,
        "controls": controls,
        "metrics": [
            {"label": "Total Bets", "value": str(len(df_recs)), "delta": None, "help": None},
            {"label": "Record", "value": f"{wins}–{losses}" if total else "N/A", "delta": None, "help": None},
            {"label": "Win Rate", "value": f"{wins / total:.1%}" if total else "—", "delta": None, "help": None},
            {
                "label": "Strong Edge Tier",
                "value": str(int((df_recs["Edge Tier"] == "STRONG").sum())),
                "delta": None,
                "help": None,
            },
        ],
        "table_heading": "All Value Bets",
        "table": {"columns": display_cols, "rows": records(table)},
        "simulator_heading": "💸 Bankroll Simulator",
        "simulator_caption": (
            "The bankroll chart is a scenario tool. "
            "It does not infer cover probability from edge points."
        ),
        "scenario_help": (
            "User-supplied scenario; spread/total point edges are not calibrated win probabilities."
        ),
        "chart": chart,
        "chart_empty_caption": (
            "No completed bets to simulate — check back once the season has results."
        ),
        "footer": True,
    }
