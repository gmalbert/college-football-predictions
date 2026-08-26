"""utils/models.py

Train, persist, load, and run inference for four prediction targets:
  - Win probability   (XGBoost binary classifier / logistic-regression fallback)
  - Point spread      (XGBoost regressor / ridge fallback)
  - Over/under total  (Ridge regression)
  - Over probability  (regularized closing-time logistic classifier)

Typical workflow:
  1.  fetch_historical.run()           # populate processed Parquet tables
  2.  feature_engine.build_feature_matrix()
  3.  models.train_all()              # trains and saves .joblib files
  4.  models.predict(home, away, …)   # inference
"""
from __future__ import annotations

import json
import os
from statistics import NormalDist
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    # A dashboard should remain readable when a restricted desktop runtime blocks
    # native XGBoost DLL loading. Inference below degrades to unavailable values.
    xgb = None
    HAS_XGB = False

from utils.feature_engine import (
    WIN_FEATURES, SPREAD_FEATURES, TOTAL_FEATURES, TOTAL_COVER_FEATURES,
)
from utils.challenger_models import (
    MarketAnchoredRegressor,
    MarketBaselineClassifier,
    MarketBaselineRegressor,
    fit_market_residual,
    walk_forward_market_residual,
)
from utils.evaluation import evaluate_release_gates, probability_metrics, regression_metrics
from utils.market import remove_vig
from utils.model_registry import create_manifest, promotion_decision, save_manifest
from utils.storage import MODELS_DIR, load_parquet, save_parquet
from utils.temporal import walk_forward_season_splits
from utils.logger import get_logger

logger = get_logger(__name__)

WIN_MODEL_PATH    = MODELS_DIR / "win_prob_model.joblib"
SPREAD_MODEL_PATH = MODELS_DIR / "spread_model.joblib"
TOTAL_MODEL_PATH  = MODELS_DIR / "total_model.joblib"
TOTAL_COVER_MODEL_PATH = MODELS_DIR / "total_cover_model.joblib"
METRICS_PATH      = MODELS_DIR / "model_metrics.json"
MODEL_VERSION     = "2.2.0"
TOTAL_COVER_C = 0.005
TOTAL_COVER_EDGE = 0.075


class Prediction(NamedTuple):
    win_prob: float           # P(home team wins), 0–1
    predicted_spread: float   # home score − away score
    predicted_total: float    # total points


# ─────────────────────────────── Elo model ─────────────────────────────────
_K_FACTOR        = 20
_HOME_ADVANTAGE  = 65
_INITIAL_ELO     = 1500
_REVERSION       = 0.33   # mean-revert 1/3 toward 1500 each off-season


class EloModel:
    """
    Simple Elo rating system for college football.
    Update ratings forward through a schedule to get pre-game win-probability
    estimates and per-game elo_diff features.
    """

    def __init__(self, k: float = _K_FACTOR, home_adv: float = _HOME_ADVANTAGE):
        self.k = k
        self.home_adv = home_adv
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.setdefault(team, _INITIAL_ELO)

    def predict(self, home: str, away: str) -> float:
        """Return P(home wins), adjusted for home-field advantage."""
        adj_home = self.get_rating(home) + self.home_adv
        return 1.0 / (1.0 + 10 ** ((self.get_rating(away) - adj_home) / 400))

    def update(self, home: str, away: str, home_won: bool) -> None:
        adj_home = self.get_rating(home) + self.home_adv
        exp_home = 1.0 / (1.0 + 10 ** ((self.get_rating(away) - adj_home) / 400))
        delta = self.k * ((1 if home_won else 0) - exp_home)
        self.ratings[home] = self.get_rating(home) + delta
        self.ratings[away] = self.get_rating(away) - delta

    def new_season(self) -> None:
        """Mean-revert all ratings toward 1500 between seasons."""
        for team in self.ratings:
            self.ratings[team] += _REVERSION * (_INITIAL_ELO - self.ratings[team])


# ─────────────────────────────── training ────────────────────────────────────

def train_all(force: bool = False) -> dict:
    """
    Train all models on the feature matrix.
    Returns a metrics dict; persists .joblib files to data_files/models/.
    If models already exist and force=False, skips retraining.
    """
    if not force and all(
        p.exists() for p in [
            WIN_MODEL_PATH, SPREAD_MODEL_PATH, TOTAL_MODEL_PATH,
            TOTAL_COVER_MODEL_PATH,
        ]
    ):
        logger.info("Models already trained — skipping (use force=True to retrain).")
        return load_metrics()

    try:
        df = load_parquet("feature_matrix", layer="features")
    except FileNotFoundError:
        logger.error(
            "Feature matrix not found. "
            "Run fetch_historical.run() + build_feature_matrix() first."
        )
        return {}

    metrics: dict = {"model_version": MODEL_VERSION, "evaluation_scope": "walk_forward_season_oos"}

    # Cardinality is a hard model contract. The previous feature artifact had
    # 4,443 duplicate game rows caused by a team-week rolling-stat join.
    if "game_id" in df.columns:
        duplicates = int(df["game_id"].duplicated().sum())
        if duplicates:
            logger.warning(f"Dropping {duplicates:,} duplicate game rows before training")
            df = df.drop_duplicates("game_id", keep="last")
    sort_columns = [c for c in ("start_date", "game_id") if c in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    # Ensure every declared feature column exists in the DataFrame.
    # Columns missing entirely (e.g. weather on free tier) are zeroed out so
    # the saved models always see a fixed-width vector at inference time too.
    all_feats = list(dict.fromkeys(
        WIN_FEATURES + SPREAD_FEATURES + TOTAL_FEATURES + TOTAL_COVER_FEATURES
    ))
    for f in all_feats:
        if f not in df.columns:
            df[f] = np.nan

    # ── Win probability ──────────────────────────────────────────────────────
    win_feats = WIN_FEATURES
    df_win = df.dropna(subset=["home_win", "season"]).copy()
    X_win  = df_win[win_feats]
    y_win  = df_win["home_win"].values.astype(int)

    win_model, win_m, win_oof = _walk_forward_classifier(
        X_win, y_win, df_win[["season", *(["start_date"] if "start_date" in df_win else [])]]
    )
    market_home = _market_home_probability(df_win)
    priced_win = np.isfinite(market_home) if market_home is not None else np.zeros(len(df_win), dtype=bool)
    win_oof[priced_win & np.isfinite(win_oof)] = market_home[priced_win & np.isfinite(win_oof)]
    win_model = MarketBaselineClassifier(
        fallback_model=win_model,
        feature_names=tuple(win_feats),
    )
    oos_mask = np.isfinite(win_oof)
    win_fold_metrics = _probability_fold_metrics(
        y_win, win_oof,
        df_win[["season", *(["start_date"] if "start_date" in df_win else [])]],
        baseline=market_home,
    )
    win_m = probability_metrics(
        y_win[oos_mask], win_oof[oos_mask],
        baseline_probabilities=market_home[oos_mask] if market_home is not None else None,
    ) | {
        "folds": win_fold_metrics, "n_samples": int(oos_mask.sum()),
        "strategy": "no_vig_market_when_available_structural_fallback",
        "market_anchored_n": int((priced_win & oos_mask).sum()),
    }
    joblib.dump(win_model, WIN_MODEL_PATH)
    metrics["win_model"] = win_m
    _ll_str = f"  log_loss={win_m['log_loss']:.4f}" if "log_loss" in win_m else ""
    logger.info(f"  win model  — brier={win_m.get('brier', '?'):.4f}{_ll_str}   oos_n={win_m['n_samples']:,}")

    # ── Spread ───────────────────────────────────────────────────────────────
    sp_feats = SPREAD_FEATURES
    df_sp    = df.dropna(subset=["home_margin", "season"]).copy()
    X_sp     = df_sp[sp_feats]
    y_sp     = df_sp["home_margin"].values

    spread_model, sp_meta, spread_oof = _walk_forward_regressor(
        X_sp, y_sp, df_sp[["season", *(["start_date"] if "start_date" in df_sp else [])]],
        model_kind="xgb" if HAS_XGB else "ridge",
    )
    sp_oos = np.isfinite(spread_oof)
    market_margin = -pd.to_numeric(df_sp.get("market_spread"), errors="coerce").to_numpy(dtype=float)
    lined_spread = np.isfinite(market_margin) & np.isfinite(spread_oof)
    spread_oof[lined_spread] = market_margin[lined_spread]
    spread_model = MarketBaselineRegressor(
        fallback_model=spread_model,
        feature_names=tuple(sp_feats),
        market_feature="market_spread",
        market_sign=-1.0,
    )
    spread_fold_metrics = _regression_fold_metrics(
        y_sp, spread_oof,
        df_sp[["season", *(["start_date"] if "start_date" in df_sp else [])]],
        baseline=market_margin,
    )
    sp_m = regression_metrics(
        y_sp[sp_oos], spread_oof[sp_oos], baseline_predictions=market_margin[sp_oos]
    ) | {
        "folds": spread_fold_metrics, "n_samples": int(sp_oos.sum()),
        "strategy": "market_consensus_when_available_structural_fallback",
        "market_anchored_n": int(lined_spread.sum()),
    }
    joblib.dump(spread_model, SPREAD_MODEL_PATH)
    metrics["spread_model"] = sp_m
    logger.info(f"  spread model — rmse={sp_m.get('rmse', '?'):.2f}   oos_n={sp_m['n_samples']:,}")

    # ── Total ────────────────────────────────────────────────────────────────
    tot_feats = TOTAL_FEATURES
    df_tot    = df.dropna(subset=["total_points", "season"]).copy()
    X_tot     = df_tot[tot_feats]
    y_tot     = df_tot["total_points"].values

    total_model, tot_meta, total_oof = _walk_forward_regressor(
        X_tot, y_tot, df_tot[["season", *(["start_date"] if "start_date" in df_tot else [])]],
        model_kind="ridge",
    )
    tot_oos = np.isfinite(total_oof)
    market_total = pd.to_numeric(df_tot.get("market_total"), errors="coerce").to_numpy(dtype=float)
    residual_oof, residual_folds = walk_forward_market_residual(
        X_tot, y_tot, market_total,
        df_tot[["season", *(["start_date"] if "start_date" in df_tot else [])]],
        kind="ridge",
    )
    residual_valid = np.isfinite(residual_oof) & np.isfinite(total_oof)
    total_oof[residual_valid] = residual_oof[residual_valid]
    residual_model, residual_shrinkage, residual_shrinkage_n = fit_market_residual(
        X_tot, y_tot, market_total,
        df_tot[["season", *(["start_date"] if "start_date" in df_tot else [])]],
        kind="ridge",
    )
    total_model = MarketAnchoredRegressor(
        residual_model=residual_model,
        fallback_model=total_model,
        feature_names=tuple(tot_feats),
        market_feature="market_total",
        market_sign=1.0,
        shrinkage=residual_shrinkage,
    )
    total_fold_metrics = _regression_fold_metrics(
        y_tot, total_oof,
        df_tot[["season", *(["start_date"] if "start_date" in df_tot else [])]],
        baseline=market_total,
    )
    tot_m = regression_metrics(
        y_tot[tot_oos], total_oof[tot_oos], baseline_predictions=market_total[tot_oos]
    ) | {
        "folds": total_fold_metrics, "n_samples": int(tot_oos.sum()),
        "strategy": "nested_oos_shrunk_market_residual_ridge",
        "residual_folds": residual_folds,
        "final_residual_shrinkage": residual_shrinkage,
        "final_residual_shrinkage_oos_n": residual_shrinkage_n,
    }
    joblib.dump(total_model, TOTAL_MODEL_PATH)
    metrics["total_model"] = tot_m
    logger.info(f"  total model  — rmse={tot_m.get('rmse', '?'):.2f}   oos_n={tot_m['n_samples']:,}")

    # ── Closing-time total side probability ────────────────────────────────
    # Pushes have no binary over/under label and are excluded from classifier
    # fitting and grading. Market movement fields are available only under the
    # explicit closing-time contract used by this model.
    df_cover = df.dropna(subset=["total_points", "market_total", "season"]).copy()
    df_cover = df_cover[
        pd.to_numeric(df_cover["total_points"], errors="coerce")
        != pd.to_numeric(df_cover["market_total"], errors="coerce")
    ].copy()
    X_cover = df_cover[TOTAL_COVER_FEATURES]
    y_cover = (
        pd.to_numeric(df_cover["total_points"], errors="coerce")
        > pd.to_numeric(df_cover["market_total"], errors="coerce")
    ).astype(int).to_numpy()
    cover_temporal = df_cover[
        ["season", *(["start_date"] if "start_date" in df_cover else [])]
    ]
    total_cover_model, cover_meta, cover_oof = _walk_forward_total_cover_classifier(
        X_cover, y_cover, cover_temporal, c=TOTAL_COVER_C
    )
    cover_valid = np.isfinite(cover_oof)
    cover_baseline = np.full(len(y_cover), 0.5, dtype=float)
    cover_metrics = probability_metrics(
        y_cover[cover_valid], cover_oof[cover_valid],
        baseline_probabilities=cover_baseline[cover_valid],
    ) | {
        "folds": _probability_fold_metrics(
            y_cover, cover_oof, cover_temporal, baseline=cover_baseline
        ),
        "n_samples": int(cover_valid.sum()),
        "strategy": _total_cover_strategy_metrics(
            df_cover, cover_oof, probability_edge=TOTAL_COVER_EDGE
        ),
        "prediction_contract": "final eligible pre-kickoff market snapshot",
        "regularization_c": TOTAL_COVER_C,
    }
    break_even = 110 / 210
    strategy_metrics = cover_metrics["strategy"]
    eligible_sides: list[str] = []
    for side in ("over", "under"):
        confirmation = strategy_metrics.get("by_season_and_side", {}).get(
            "2025", {}
        ).get(side, {})
        season_rows = [
            values.get(side, {})
            for values in strategy_metrics.get("by_season_and_side", {}).values()
            if values.get(side, {}).get("n", 0) > 0
        ]
        if (
            confirmation.get("n", 0) >= 50
            and confirmation.get("wilson_95_low", 0.0) > break_even
            and season_rows
            and all(row.get("win_rate", 0.0) > break_even for row in season_rows)
        ):
            eligible_sides.append(side)
    retrospective_pass = bool(
        cover_metrics.get("brier", 1.0) < cover_metrics.get("baseline_brier", 0.0)
        and eligible_sides
    )
    cover_metrics["strategy_release"] = {
        "status": "shadow" if retrospective_pass else "hold",
        "retrospective_gates_passed": retrospective_pass,
        "eligible_sides": eligible_sides,
        "prospective_clv_passed": False,
        "reason": (
            "Retrospective confirmation passed; collect 2026 closing-line value before live promotion."
            if retrospective_pass else
            "Retrospective probability/confirmation gates did not pass."
        ),
    }
    joblib.dump(total_cover_model, TOTAL_COVER_MODEL_PATH)
    metrics["total_cover_model"] = cover_metrics
    logger.info(
        "  total side — brier=%.4f  selected=%d  win=%.1f%%  status=%s",
        cover_metrics.get("brier", float("nan")),
        cover_metrics["strategy"].get("n", 0),
        100 * cover_metrics["strategy"].get("win_rate", 0.0),
        cover_metrics["strategy_release"]["status"],
    )

    # ── ATS backtest ────────────────────────────────────────────────────────
    metrics["ats"] = _ats_record_oos(df_sp, spread_oof)
    logger.info(
        f"  ATS record — {metrics['ats']['wins']}W "
        f"{metrics['ats']['losses']}L  "
        f"({metrics['ats']['pct']:.1%})"
    )

    # Persist only walk-forward predictions for honest diagnostics. The final
    # full-data models are for future inference and must never grade themselves.
    backtest_columns = [
        column for column in (
            "game_id", "season", "week", "start_date", "home_win", "home_margin",
            "total_points", "market_spread", "market_total", "home_moneyline",
            "away_moneyline", "market_spread_open", "market_spread_move",
            "market_total_open", "market_total_move",
            "home_team", "away_team",
        ) if column in df.columns
    ]
    backtest = df[backtest_columns].drop_duplicates("game_id").copy()
    backtest["win_prob_oos"] = backtest["game_id"].map(
        dict(zip(df_win["game_id"], win_oof))
    )
    backtest["predicted_spread_oos"] = backtest["game_id"].map(
        dict(zip(df_sp["game_id"], spread_oof))
    )
    backtest["predicted_total_oos"] = backtest["game_id"].map(
        dict(zip(df_tot["game_id"], total_oof))
    )
    backtest["total_over_prob_oos"] = backtest["game_id"].map(
        dict(zip(df_cover["game_id"], cover_oof))
    )
    save_parquet(backtest, "model_backtest", layer="features")

    metrics["release_gates"] = evaluate_release_gates(metrics)
    metrics["release_decision"] = promotion_decision(metrics["release_gates"])
    git_sha = os.environ.get("GITHUB_SHA")
    manifest_specs = (
        ("win_probability", df_win, WIN_FEATURES, win_m, "P(home win), calibrated probability in [0,1]"),
        ("home_margin", df_sp, SPREAD_FEATURES, sp_m, "home_score - away_score; positive means home by N"),
        ("game_total", df_tot, TOTAL_FEATURES, tot_m, "home_score + away_score in points"),
        (
            "total_over_probability", df_cover, TOTAL_COVER_FEATURES, cover_metrics,
            "P(game total exceeds current market total) at final eligible pre-kickoff snapshot",
        ),
    )
    manifest_paths: dict[str, str] = {}
    for name, training_frame, features, model_metrics, contract in manifest_specs:
        manifest = create_manifest(
            model_name=name,
            version=MODEL_VERSION,
            training_frame=training_frame,
            features=features,
            parameters={
                "validation": "expanding-season-walk-forward",
                "min_train_seasons": 2,
                "imputation": (
                    "native_missing"
                    if HAS_XGB and name in {"win_probability", "home_margin"}
                    else "median_with_indicator"
                ),
            },
            metrics=model_metrics,
            prediction_contract=contract,
            git_sha=git_sha,
        )
        manifest_path = MODELS_DIR / f"{name}_manifest.json"
        save_manifest(manifest, manifest_path)
        manifest_paths[name] = manifest_path.name
    metrics["artifact_manifests"] = manifest_paths

    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


# ─────────────────────────────── inference ───────────────────────────────────

def predict(
    home: str,
    away: str,
    season: int,
    week: int,
) -> Prediction | None:
    """
    Run all three models for a single game identified by home/away/season/week.
    Returns None if models are not yet trained or the game row is missing.
    """
    if not models_trained():
        return None

    try:
        df = load_parquet("feature_matrix", layer="features")
    except FileNotFoundError:
        return None

    mask = (
        (df["home_team"] == home)
        & (df["away_team"] == away)
        & (df["season"] == season)
        & (df["week"] == week)
    )
    if not mask.any():
        return None
    row = df[mask].iloc[0]
    return _predict_row(row)


def predict_row(row: pd.Series) -> Prediction | None:
    """Run all models on a pre-built feature row (e.g., for a future game)."""
    if not models_trained():
        return None
    return _predict_row(row)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add model prediction columns to a feature-matrix DataFrame slice.
    Returns df with new columns: win_prob, predicted_spread, predicted_total.
    """
    if not models_trained():
        return df
    def _load(path):
        try:
            return joblib.load(path)
        except Exception:
            return None

    win_m = _load(WIN_MODEL_PATH)
    spread_m = _load(SPREAD_MODEL_PATH)
    total_m = _load(TOTAL_MODEL_PATH)
    total_cover_m = _load(TOTAL_COVER_MODEL_PATH)

    # Add absent columns as NaN. XGBoost learns a missing-value direction and
    # sklearn pipelines apply their persisted median/indicator transformer.
    df = df.copy()
    for f in list(dict.fromkeys(
        WIN_FEATURES + SPREAD_FEATURES + TOTAL_FEATURES + TOTAL_COVER_FEATURES
    )):
        if f not in df.columns:
            df[f] = np.nan

    try:
        X = df[WIN_FEATURES]
        df["win_prob"] = _clf_predict_proba(win_m, X)
    except Exception:
        df["win_prob"] = float("nan")

    try:
        X = df[SPREAD_FEATURES]
        df["predicted_spread"] = _reg_predict(spread_m, X)
    except Exception:
        df["predicted_spread"] = float("nan")

    try:
        X = df[TOTAL_FEATURES]
        df["predicted_total"] = _reg_predict(total_m, X)
    except Exception:
        df["predicted_total"] = float("nan")

    try:
        X = df[TOTAL_COVER_FEATURES]
        df["total_over_prob"] = _clf_predict_proba(total_cover_m, X)
    except Exception:
        df["total_over_prob"] = float("nan")

    return df


def predict_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return honest predictions for mixed historical/upcoming UI slices.

    Completed games receive their saved season walk-forward predictions. Only
    unplayed games are scored by the final model fitted on all available history.
    ``prediction_scope`` makes that distinction visible to callers.
    """
    result = df.copy()
    prediction_columns = [
        "win_prob", "predicted_spread", "predicted_total", "total_over_prob"
    ]
    for column in prediction_columns:
        result[column] = np.nan
    result["prediction_scope"] = "unavailable"

    if "home_margin" in result.columns:
        completed = pd.to_numeric(result["home_margin"], errors="coerce").notna()
    elif {"home_score", "away_score"}.issubset(result.columns):
        completed = result["home_score"].notna() & result["away_score"].notna()
    else:
        completed = pd.Series(False, index=result.index)

    upcoming = ~completed
    if upcoming.any() and models_trained():
        scored = predict_batch(result.loc[upcoming])
        result.loc[upcoming, prediction_columns] = scored[prediction_columns].to_numpy()
        result.loc[upcoming, "prediction_scope"] = "future_full_fit"

    if completed.any() and "game_id" in result.columns:
        try:
            backtest = load_parquet("model_backtest", layer="features")
        except FileNotFoundError:
            backtest = pd.DataFrame()
        if not backtest.empty and "game_id" in backtest.columns:
            oos_columns = {
                "win_prob_oos": "win_prob",
                "predicted_spread_oos": "predicted_spread",
                "predicted_total_oos": "predicted_total",
                "total_over_prob_oos": "total_over_prob",
            }
            available = ["game_id", *[c for c in oos_columns if c in backtest.columns]]
            lookup = backtest[available].drop_duplicates("game_id").set_index("game_id")
            for source, target in oos_columns.items():
                if source in lookup.columns:
                    mapped = result.loc[completed, "game_id"].map(lookup[source])
                    result.loc[completed, target] = mapped.to_numpy()
            has_oos = result.loc[completed, prediction_columns].notna().any(axis=1)
            result.loc[has_oos.index[has_oos], "prediction_scope"] = "walk_forward_oos"
    return result


def load_models() -> dict:
    def _safe(path):
        try:
            return joblib.load(path) if path.exists() else None
        except Exception:
            return None
    return {
        "win": _safe(WIN_MODEL_PATH),
        "spread": _safe(SPREAD_MODEL_PATH),
        "total": _safe(TOTAL_MODEL_PATH),
        "total_cover": _safe(TOTAL_COVER_MODEL_PATH),
    }


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    return {}


def models_trained() -> bool:
    return all(
        p.exists() for p in [
            WIN_MODEL_PATH, SPREAD_MODEL_PATH, TOTAL_MODEL_PATH,
            TOTAL_COVER_MODEL_PATH,
        ]
    )


# ─────────────────────────────── private helpers ─────────────────────────────

def _get_val(row: pd.Series, col: str, default: float = 0.0) -> float:
    v = row.get(col, default)
    return float(v) if pd.notna(v) else default


def _predict_row(row: pd.Series) -> Prediction:
    win_m    = joblib.load(WIN_MODEL_PATH)
    spread_m = joblib.load(SPREAD_MODEL_PATH)
    total_m  = joblib.load(TOTAL_MODEL_PATH)

    w_x = pd.DataFrame([{f: _get_val(row, f, np.nan) for f in WIN_FEATURES}])
    s_x = pd.DataFrame([{f: _get_val(row, f, np.nan) for f in SPREAD_FEATURES}])
    t_x = pd.DataFrame([{f: _get_val(row, f, np.nan) for f in TOTAL_FEATURES}])

    return Prediction(
        win_prob=float(_clf_predict_proba(win_m, w_x)[0]),
        predicted_spread=float(_reg_predict(spread_m, s_x)[0]),
        predicted_total=float(_reg_predict(total_m, t_x)[0]),
    )


def _clf_predict_proba(model, X):
    if HAS_XGB and isinstance(model, xgb.Booster):
        if hasattr(X, "columns"):
            dmat = xgb.DMatrix(X.values, feature_names=[str(c) for c in X.columns])
        else:
            dmat = xgb.DMatrix(X)
        return model.predict(dmat)
    return model.predict_proba(X)[:, 1]


def _reg_predict(model, X):
    if HAS_XGB and isinstance(model, xgb.Booster):
        if hasattr(X, "columns"):
            dmat = xgb.DMatrix(X.values, feature_names=[str(c) for c in X.columns])
        else:
            dmat = xgb.DMatrix(X)
        return model.predict(dmat)
    return model.predict(X)


def _base_classifier_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000)),
    ])


def _base_ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])


def _base_total_cover_pipeline(c: float = TOTAL_COVER_C) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=c, max_iter=2000)),
    ])


def _walk_forward_total_cover_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    temporal: pd.DataFrame,
    *,
    c: float,
):
    folds = list(walk_forward_season_splits(temporal, min_train_seasons=2))
    if not folds:
        raise ValueError("total-side model requires at least three seasons")
    oof = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict] = []
    for fold in folds:
        model = _base_total_cover_pipeline(c).fit(
            X.iloc[fold.train_index], y[fold.train_index]
        )
        prediction = model.predict_proba(X.iloc[fold.test_index])[:, 1]
        oof[fold.test_index] = prediction
        fold_rows.append(
            {
                "fold": fold.fold,
                "train_seasons": list(fold.train_seasons),
                "test_season": fold.test_season,
                **probability_metrics(
                    y[fold.test_index], prediction,
                    baseline_probabilities=np.full(len(prediction), 0.5),
                ),
            }
        )
    final = _base_total_cover_pipeline(c).fit(X, y)
    return final, {"folds": fold_rows}, oof


def _wilson_interval(
    wins: int, n: int, confidence: float = 0.95
) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = wins / n
    denominator = 1 + z * z / n
    center = proportion + z * z / (2 * n)
    radius = z * np.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * n)) / n
    )
    return float((center - radius) / denominator), float((center + radius) / denominator)


def _total_cover_strategy_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    probability_edge: float,
    assumed_odds: float = -110.0,
) -> dict:
    probability = np.asarray(probabilities, dtype=float)
    target = (
        pd.to_numeric(frame["total_points"], errors="coerce")
        > pd.to_numeric(frame["market_total"], errors="coerce")
    ).to_numpy(bool)
    selected = np.isfinite(probability) & (np.abs(probability - 0.5) >= probability_edge)
    wins = np.where(probability[selected] > 0.5, target[selected], ~target[selected])

    def summarize(mask: np.ndarray) -> dict:
        chosen = selected & mask
        outcomes = np.where(probability[chosen] > 0.5, target[chosen], ~target[chosen])
        n = int(len(outcomes))
        win_count = int(outcomes.sum())
        lower, upper = _wilson_interval(win_count, n)
        decimal_profit = 100 / abs(assumed_odds) if assumed_odds < 0 else assumed_odds / 100
        profit_units = win_count * decimal_profit - (n - win_count)
        return {
            "n": n,
            "wins": win_count,
            "losses": n - win_count,
            "win_rate": win_count / n if n else 0.0,
            "wilson_95_low": lower,
            "wilson_95_high": upper,
            "flat_stake_roi_at_minus_110": profit_units / n if n else 0.0,
        }

    all_rows = np.ones(len(frame), dtype=bool)
    result = summarize(all_rows)
    seasons = pd.to_numeric(frame["season"], errors="coerce").to_numpy(float)
    result.update(
        {
            "probability_edge_threshold": probability_edge,
            "minimum_selected_probability": 0.5 + probability_edge,
            "assumed_odds": assumed_odds,
            "break_even_win_rate": abs(assumed_odds) / (abs(assumed_odds) + 100),
            "by_season": {
                str(int(season)): summarize(seasons == season)
                for season in sorted(set(seasons[np.isfinite(seasons)]))
            },
            "by_side": {
                "over": summarize(probability > 0.5),
                "under": summarize(probability < 0.5),
            },
            "by_season_and_side": {
                str(int(season)): {
                    "over": summarize((seasons == season) & (probability > 0.5)),
                    "under": summarize((seasons == season) & (probability < 0.5)),
                }
                for season in sorted(set(seasons[np.isfinite(seasons)]))
            },
            "development_note": (
                "C and threshold developed on 2021-2024; 2025 is the retrospective confirmation season."
            ),
        }
    )
    return result


def _probability_fold_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    temporal: pd.DataFrame,
    *,
    baseline: np.ndarray | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for fold in walk_forward_season_splits(temporal, min_train_seasons=2):
        test = fold.test_index[np.isfinite(prediction[fold.test_index])]
        if not len(test):
            continue
        rows.append(
            {
                "fold": fold.fold,
                "train_seasons": list(fold.train_seasons),
                "test_season": fold.test_season,
                **probability_metrics(
                    y[test], prediction[test],
                    baseline_probabilities=baseline[test] if baseline is not None else None,
                ),
            }
        )
    return rows


def _regression_fold_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    temporal: pd.DataFrame,
    *,
    baseline: np.ndarray | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for fold in walk_forward_season_splits(temporal, min_train_seasons=2):
        test = fold.test_index[np.isfinite(prediction[fold.test_index])]
        if not len(test):
            continue
        rows.append(
            {
                "fold": fold.fold,
                "train_seasons": list(fold.train_seasons),
                "test_season": fold.test_season,
                **regression_metrics(
                    y[test], prediction[test],
                    baseline_predictions=baseline[test] if baseline is not None else None,
                ),
            }
        )
    return rows


def _walk_forward_classifier(X: pd.DataFrame, y: np.ndarray, temporal: pd.DataFrame):
    folds = list(walk_forward_season_splits(temporal, min_train_seasons=2))
    if not folds:
        raise ValueError("win model requires at least three seasons for walk-forward evaluation")
    oof = np.full(len(y), np.nan, dtype=float)
    fold_metrics: list[dict] = []
    best_rounds: list[int] = []
    for fold in folds:
        if HAS_XGB:
            dtrain = xgb.DMatrix(X.iloc[fold.train_index], label=y[fold.train_index])
            dtest = xgb.DMatrix(X.iloc[fold.test_index], label=y[fold.test_index])
            model = xgb.train(
                {
                    "objective": "binary:logistic", "eval_metric": "logloss",
                    "max_depth": 5, "eta": 0.04, "subsample": 0.8,
                    "colsample_bytree": 0.8, "min_child_weight": 8,
                    "seed": 42, "verbosity": 0,
                },
                dtrain,
                num_boost_round=600,
                evals=[(dtest, "validation")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            prediction = model.predict(dtest)
            best_rounds.append(int(getattr(model, "best_iteration", 399)) + 1)
        else:
            model = _base_classifier_pipeline()
            model.fit(X.iloc[fold.train_index], y[fold.train_index])
            prediction = model.predict_proba(X.iloc[fold.test_index])[:, 1]
        oof[fold.test_index] = prediction
        fold_metrics.append(
            {
                "fold": fold.fold,
                "train_seasons": list(fold.train_seasons),
                "test_season": fold.test_season,
                **probability_metrics(y[fold.test_index], prediction),
            }
        )

    if HAS_XGB:
        rounds = int(np.median(best_rounds)) if best_rounds else 400
        final = xgb.train(
            {
                "objective": "binary:logistic", "eval_metric": "logloss",
                "max_depth": 5, "eta": 0.04, "subsample": 0.8,
                "colsample_bytree": 0.8, "min_child_weight": 8,
                "seed": 42, "verbosity": 0,
            },
            xgb.DMatrix(X, label=y),
            num_boost_round=rounds,
            verbose_eval=False,
        )
    else:
        final = _base_classifier_pipeline().fit(X, y)
    return final, {"folds": fold_metrics}, oof


def _walk_forward_regressor(
    X: pd.DataFrame,
    y: np.ndarray,
    temporal: pd.DataFrame,
    *,
    model_kind: str,
):
    folds = list(walk_forward_season_splits(temporal, min_train_seasons=2))
    if not folds:
        raise ValueError("regression models require at least three seasons for walk-forward evaluation")
    oof = np.full(len(y), np.nan, dtype=float)
    fold_metrics: list[dict] = []
    best_rounds: list[int] = []
    for fold in folds:
        if model_kind == "xgb" and HAS_XGB:
            dtrain = xgb.DMatrix(X.iloc[fold.train_index], label=y[fold.train_index])
            dtest = xgb.DMatrix(X.iloc[fold.test_index], label=y[fold.test_index])
            model = xgb.train(
                {
                    "objective": "reg:squarederror", "eval_metric": "rmse",
                    "max_depth": 5, "eta": 0.04, "subsample": 0.8,
                    "colsample_bytree": 0.8, "min_child_weight": 8,
                    "seed": 42, "verbosity": 0,
                },
                dtrain,
                num_boost_round=600,
                evals=[(dtest, "validation")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            prediction = model.predict(dtest)
            best_rounds.append(int(getattr(model, "best_iteration", 349)) + 1)
        else:
            model = _base_ridge_pipeline()
            model.fit(X.iloc[fold.train_index], y[fold.train_index])
            prediction = model.predict(X.iloc[fold.test_index])
        oof[fold.test_index] = prediction
        fold_metrics.append(
            {
                "fold": fold.fold,
                "train_seasons": list(fold.train_seasons),
                "test_season": fold.test_season,
                **regression_metrics(y[fold.test_index], prediction),
            }
        )

    if model_kind == "xgb" and HAS_XGB:
        rounds = int(np.median(best_rounds)) if best_rounds else 350
        final = xgb.train(
            {
                "objective": "reg:squarederror", "eval_metric": "rmse",
                "max_depth": 5, "eta": 0.04, "subsample": 0.8,
                "colsample_bytree": 0.8, "min_child_weight": 8,
                "seed": 42, "verbosity": 0,
            },
            xgb.DMatrix(X, label=y),
            num_boost_round=rounds,
            verbose_eval=False,
        )
    else:
        final = _base_ridge_pipeline().fit(X, y)
    return final, {"folds": fold_metrics}, oof


def _market_home_probability(frame: pd.DataFrame) -> np.ndarray | None:
    if "market_home_prob" in frame.columns:
        consensus = pd.to_numeric(
            frame["market_home_prob"], errors="coerce"
        ).to_numpy(dtype=float)
        if np.isfinite(consensus).any():
            return consensus
    if not {"home_moneyline", "away_moneyline"}.issubset(frame.columns):
        return None
    probabilities = np.full(len(frame), np.nan, dtype=float)
    for position, (home_odds, away_odds) in enumerate(
        zip(frame["home_moneyline"], frame["away_moneyline"])
    ):
        if pd.isna(home_odds) or pd.isna(away_odds):
            continue
        try:
            probabilities[position] = remove_vig([float(home_odds), float(away_odds)])[0]
        except (TypeError, ValueError):
            continue
    return probabilities


def _ats_record_oos(
    frame: pd.DataFrame, predictions: np.ndarray, min_edge_points: float = 1.0
) -> dict:
    actual = pd.to_numeric(frame["home_margin"], errors="coerce").to_numpy(dtype=float)
    line = pd.to_numeric(frame.get("market_spread"), errors="coerce").to_numpy(dtype=float)
    prediction = np.asarray(predictions, dtype=float)
    priced = np.isfinite(actual) & np.isfinite(line) & np.isfinite(prediction)
    settled = actual + line
    edge = prediction + line
    actionable = priced & (np.abs(edge) >= min_edge_points)
    valid = actionable & (settled != 0)
    correct = ((prediction[valid] + line[valid]) > 0) == (settled[valid] > 0)
    return {
        "wins": int(correct.sum()),
        "losses": int((~correct).sum()),
        "pushes": int(np.sum((actual + line == 0) & np.isfinite(prediction))),
        "pct": round(float(correct.mean()), 4) if len(correct) else 0.0,
        "n": int(len(correct)),
        "abstentions": int((priced & ~actionable).sum()),
        "minimum_edge_points": min_edge_points,
        "scope": "walk_forward_season_oos_actionable_lined_games",
    }


def _tscv(X, y, n_splits: int = 5):
    return list(TimeSeriesSplit(n_splits=n_splits).split(X))


def _train_logreg(X, y):
    pipe = _base_classifier_pipeline()
    briers = []
    for tr, va in _tscv(X, y):
        pipe.fit(X[tr], y[tr])
        briers.append(brier_score_loss(y[va], pipe.predict_proba(X[va])[:, 1]))
    pipe.fit(X, y)
    return pipe, {"brier": float(np.mean(briers))}


def _train_ridge(X, y):
    pipe = _base_ridge_pipeline()
    rmses = []
    for tr, va in _tscv(X, y):
        pipe.fit(X[tr], y[tr])
        rmses.append(float(np.sqrt(np.mean((pipe.predict(X[va]) - y[va]) ** 2))))
    pipe.fit(X, y)
    return pipe, {"rmse": float(np.mean(rmses))}


def _train_xgb_clf(X, y):
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 5, "verbosity": 0,
        "seed": 42,
    }
    briers, lls = [], []
    for tr, va in _tscv(X, y):
        dtrain = xgb.DMatrix(X[tr], label=y[tr])
        dval   = xgb.DMatrix(X[va],  label=y[va])
        m = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dval, "val")], early_stopping_rounds=50,
                      verbose_eval=False)
        preds = m.predict(dval)
        briers.append(brier_score_loss(y[va], preds))
        lls.append(float(log_loss(y[va], preds)))
    final = xgb.train(params, xgb.DMatrix(X, label=y),
                      num_boost_round=400, verbose_eval=False)
    return final, {"brier": float(np.mean(briers)), "log_loss": float(np.mean(lls))}


def _train_xgb_reg(X, y):
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 5, "verbosity": 0,
        "seed": 42,
    }
    rmses = []
    for tr, va in _tscv(X, y):
        dtrain = xgb.DMatrix(X[tr], label=y[tr])
        dval   = xgb.DMatrix(X[va],  label=y[va])
        m = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dval, "val")], early_stopping_rounds=40,
                      verbose_eval=False)
        preds = m.predict(dval)
        rmses.append(float(np.sqrt(np.mean((preds - y[va]) ** 2))))
    final = xgb.train(params, xgb.DMatrix(X, label=y),
                      num_boost_round=350, verbose_eval=False)
    return final, {"rmse": float(np.mean(rmses))}


def _ats_record(df: pd.DataFrame, model, feat_cols: list[str]) -> dict:
    """Compute against-the-spread win rate on the training set (diagnostic only)."""
    avail = [f for f in feat_cols if f in df.columns]
    sub   = df.dropna(subset=avail + ["home_margin", "market_spread"])
    if sub.empty:
        return {"wins": 0, "losses": 0, "pct": 0.0}
    X      = sub[avail].fillna(0).values
    preds  = _reg_predict(model, X)
    actual = sub["home_margin"].values
    book   = sub["market_spread"].values   # negative = home favored
    cover  = actual > -book               # did home cover?
    picked_home_cover = preds > -book     # compare margin forecast to implied margin
    correct = int(np.sum(picked_home_cover == cover))
    total   = len(cover)
    return {
        "wins": correct,
        "losses": total - correct,
        "pct": round(correct / total, 4) if total else 0.0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train prediction models")
    parser.add_argument("--force", action="store_true", help="Retrain even if models exist")
    args = parser.parse_args()
    result = train_all(force=args.force)
    print("Metrics:", result)
