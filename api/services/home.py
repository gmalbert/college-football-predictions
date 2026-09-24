"""Home page — mirrors ``predictions.py::home_page``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from api.columns import FEATURE_MATRIX_COLUMNS
from api.data import DATA_DIR, parquet, read_json
from api.jsonutil import fmt_int, fmt_pct, fmt_pct1, jsonable
from utils.betting import generate_spread_pick, generate_total_pick
from utils.release import load_current_release

LOGO_PATH = Path(DATA_DIR).parent / "data_files" / "logo.png"

try:  # pragma: no cover - mirrors the Streamlit fallback for locked runtimes
    from utils.model_artifacts import attach_predictions, load_metrics, models_trained

    MODEL_RUNTIME_AVAILABLE = True
except Exception:  # noqa: BLE001 - parity with the Streamlit guard
    MODEL_RUNTIME_AVAILABLE = False
    import json as _json

    from utils.storage import MODELS_DIR

    def load_metrics() -> dict:  # type: ignore[misc]
        path = MODELS_DIR / "model_metrics.json"
        return _json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def models_trained() -> bool:  # type: ignore[misc]
        return False

    def attach_predictions(frame: pd.DataFrame) -> pd.DataFrame:  # type: ignore[misc]
        return frame


def _load_dataset() -> pd.DataFrame:
    try:
        return parquet(
            "feature_matrix", layer="features", columns=FEATURE_MATRIX_COLUMNS
        )
    except FileNotFoundError:
        return pd.DataFrame()


def _load_summary() -> pd.DataFrame:
    """Return the upcoming games with predictions attached.

    The upstream feature matrix is ~21k rows x 150 columns and only a handful
    of rows are actually upcoming. Subsetting *before* copying keeps this from
    materialising a second full copy of the artifact — which measured at
    +128 MB of resident memory in scripts/measure_memory.py.
    """
    full = _load_dataset()
    if full.empty:
        return full

    try:
        if "home_margin" in full.columns:
            upcoming = pd.to_numeric(full["home_margin"], errors="coerce").isna()
        elif {"home_score", "away_score"}.issubset(full.columns):
            upcoming = full["home_score"].isna() | full["away_score"].isna()
        else:
            upcoming = pd.Series(False, index=full.index)

        if "start_date" in full.columns:
            starts = pd.to_datetime(full["start_date"], utc=True, errors="coerce")
            now = pd.Timestamp.now(tz="UTC")
            upcoming = upcoming & (starts.isna() | (starts >= now - pd.Timedelta(hours=6)))

        subset = full.loc[upcoming].copy()
        if subset.empty or not models_trained():
            return subset
        # Predictions come from the pipeline artifact, not from live inference.
        return attach_predictions(subset)
    except (KeyError, TypeError):
        return pd.DataFrame()


def _edge_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    if {"predicted_spread", "market_spread"}.issubset(frame.columns):
        frame["spread_edge"] = (
            frame["predicted_spread"] + frame["market_spread"]
        ).abs()
        columns.append("spread_edge")
    if {"predicted_total", "market_total"}.issubset(frame.columns):
        frame["total_edge"] = (
            frame["predicted_total"] - frame["market_total"]
        ).abs()
        columns.append("total_edge")
    return columns


def _upcoming_deltas() -> dict:
    df_all = _load_summary()
    if df_all.empty:
        return {"items": [], "caption": "No upcoming prediction data is currently published."}

    edge_columns = _edge_columns(df_all)
    if not edge_columns:
        return {"items": [], "caption": "No upcoming games with a positive model edge are available."}

    df_all["edge"] = df_all[edge_columns].max(axis=1, skipna=True)
    df_all["edge_market"] = "—"
    if "total_edge" in edge_columns:
        total_is_best = df_all["total_edge"].notna()
        if "spread_edge" in edge_columns:
            total_is_best &= (
                ~df_all["spread_edge"].notna()
                | (df_all["total_edge"] >= df_all["spread_edge"])
            )
        df_all.loc[total_is_best, "edge_market"] = "O/U"
    if "spread_edge" in edge_columns:
        if "total_edge" in edge_columns:
            spread_is_best = df_all["spread_edge"].notna() & (
                ~df_all["total_edge"].notna()
                | (df_all["spread_edge"] > df_all["total_edge"])
            )
        else:
            spread_is_best = df_all["spread_edge"].notna()
        df_all.loc[spread_is_best, "edge_market"] = "Spread"

    top = (
        df_all[df_all["edge"] > 0]
        .dropna(subset=["edge"])
        .nlargest(3, "edge")
    )

    items = []
    for _, row in top.iterrows():
        wp = row.get("win_prob", float("nan"))
        edge_market = row.get("edge_market", "Edge")
        if (
            edge_market == "O/U"
            and pd.notna(row.get("predicted_total"))
            and pd.notna(row.get("market_total"))
        ):
            recommendation = generate_total_pick(
                row["home_team"], row["away_team"],
                float(row["predicted_total"]), float(row["market_total"]),
                game_id=int(row["game_id"]) if pd.notna(row.get("game_id")) else None,
            )
        elif (
            edge_market == "Spread"
            and pd.notna(row.get("predicted_spread"))
            and pd.notna(row.get("market_spread"))
        ):
            recommendation = generate_spread_pick(
                row["home_team"], row["away_team"],
                float(row["predicted_spread"]), float(row["market_spread"]),
                game_id=int(row["game_id"]) if pd.notna(row.get("game_id")) else None,
            )
        else:
            recommendation = None

        summary = f"Wk {int(row['week'])} · {edge_market} edge **{row['edge']:.1f} pts**"
        if recommendation is not None:
            summary += f" · Bet **{recommendation.pick}**"
        if pd.notna(wp):
            summary += f" · Win prob {wp:.0%}"
        items.append(
            {
                "title": f"{row['away_team']} @ {row['home_team']}",
                "summary": summary,
            }
        )

    if not items:
        return {"items": [], "caption": "No upcoming games with a positive model edge are available."}
    return {"items": items, "caption": None}


def _accuracy_metrics(metrics: dict) -> dict:
    if not metrics:
        return {"metrics": [], "caption": "Model evaluation metrics are not currently published."}
    win_m = metrics.get("win_model", {})
    spread_m = metrics.get("spread_model", {})
    ats_m = metrics.get("ats", {})
    return {
        "caption": None,
        "metrics": [
            {
                "label": "Brier Score",
                "value": f"{win_m.get('brier', 0):.4f}",
                "delta": None,
                "help": "Lower is better; < 0.20 is solid",
            },
            {
                "label": "Spread RMSE",
                "value": f"{spread_m.get('rmse', 0):.2f} pts",
                "delta": None,
                "help": None,
            },
            {
                "label": "OOS ATS Win %",
                "value": fmt_pct1(ats_m.get("pct", 0)),
                "delta": None,
                "help": "Walk-forward only; 52.4% breaks even at -110",
            },
            {
                "label": "ATS Record",
                "value": f"{ats_m.get('wins', 0)}‑{ats_m.get('losses', 0)}",
                "delta": None,
                "help": None,
            },
        ],
    }


def _dataset_metrics(df_dataset: pd.DataFrame) -> dict:
    if df_dataset.empty:
        return {"metrics": [], "caption": "No dataset is currently published."}
    seasons = sorted(df_dataset["season"].dropna().unique())
    n_games = len(df_dataset)
    n_teams = len(
        set(
            df_dataset["home_team"].dropna().tolist()
            + df_dataset["away_team"].dropna().tolist()
        )
    )
    return {
        "caption": None,
        "metrics": [
            {"label": "Games", "value": f"{n_games:,}", "delta": None, "help": None},
            {"label": "Teams", "value": f"{n_teams:,}", "delta": None, "help": None},
            {
                "label": "Seasons",
                "value": f"{seasons[0]}–{seasons[-1]}",
                "delta": f"{len(seasons)} seasons in dataset",
                "help": None,
            },
            {
                "label": "Model",
                "value": "XGBoost + Ridge" if models_trained() else "Not trained",
                "delta": None,
                "help": None,
            },
        ],
    }


def build_home() -> dict:
    """Return the complete home-page payload."""
    metrics = load_metrics()
    release = metrics.get("release_decision", {}) if metrics else {}
    release_metadata = load_current_release()

    warnings: list[str] = []
    if metrics and release.get("decision") != "promote":
        failed = ", ".join(release.get("failed_gates", [])) or "release gates unavailable"
        warnings.append(
            f"Model release status: HOLD ({failed}). Forecasts are shown for research; "
            "Best Bets are published as provisional research output and are not validated recommendations."
        )

    infos: list[str] = []
    if not MODEL_RUNTIME_AVAILABLE:
        infos.append(
            "Native model inference is unavailable in this local runtime; "
            "saved evaluation evidence remains available."
        )

    captions: list[str] = []
    if release_metadata:
        captions.append(
            f"Artifact release {release_metadata.get('release_id', 'unknown')[:12]} · "
            f"{release_metadata.get('generated_at', 'unknown')}"
        )

    return {
        "page": "home",
        "title": "🏈 College Football Predictions",
        "logo": "/api/logo" if LOGO_PATH.exists() else None,
        "warnings": warnings,
        "infos": infos,
        "captions": captions,
        "columns": [
            {
                "heading": "Upcoming Model Deltas",
                "kind": "deltas",
                **_upcoming_deltas(),
            },
            {
                "heading": "📐 Model Accuracy",
                "kind": "metrics",
                # predictions.py calls st.metric() repeatedly inside one
                # column, so these stack vertically rather than sitting in a
                # st.columns() row.
                "layout": "stacked",
                **_accuracy_metrics(metrics),
            },
            {
                "heading": "📊 Dataset",
                "kind": "metrics",
                "layout": "stacked",
                **_dataset_metrics(_load_dataset()),
            },
        ],
        "footer": True,
        "meta": jsonable({"metrics_available": bool(metrics)}),
    }
