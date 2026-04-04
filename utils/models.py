"""utils/models.py

Train, persist, load, and run inference for three prediction targets:
  - Win probability   (XGBoost binary classifier / logistic-regression fallback)
  - Point spread      (XGBoost regressor / ridge fallback)
  - Over/under total  (Ridge regression)

Typical workflow:
  1.  fetch_historical.run()           # populate processed Parquet tables
  2.  feature_engine.build_feature_matrix()
  3.  models.train_all()              # trains and saves .joblib files
  4.  models.predict(home, away, …)   # inference
"""
from __future__ import annotations

import json
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from utils.feature_engine import WIN_FEATURES, SPREAD_FEATURES, TOTAL_FEATURES
from utils.storage import MODELS_DIR, load_parquet, save_parquet
from utils.logger import get_logger

logger = get_logger(__name__)

WIN_MODEL_PATH    = MODELS_DIR / "win_prob_model.joblib"
SPREAD_MODEL_PATH = MODELS_DIR / "spread_model.joblib"
TOTAL_MODEL_PATH  = MODELS_DIR / "total_model.joblib"
METRICS_PATH      = MODELS_DIR / "model_metrics.json"


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
        p.exists() for p in [WIN_MODEL_PATH, SPREAD_MODEL_PATH, TOTAL_MODEL_PATH]
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

    metrics: dict = {}

    # ── Win probability ──────────────────────────────────────────────────────
    win_feats = [f for f in WIN_FEATURES if f in df.columns]
    df_win = df.dropna(subset=win_feats + ["home_win"])
    X_win  = df_win[win_feats].values
    y_win  = df_win["home_win"].values.astype(int)

    if HAS_XGB:
        win_model, win_m = _train_xgb_clf(X_win, y_win)
    else:
        win_model, win_m = _train_logreg(X_win, y_win)
    joblib.dump(win_model, WIN_MODEL_PATH)
    metrics["win_model"] = {**win_m, "n_samples": int(len(y_win))}
    _ll_str = f"  log_loss={win_m['log_loss']:.4f}" if "log_loss" in win_m else ""
    logger.info(f"  win model  — brier={win_m.get('brier', '?'):.4f}{_ll_str}   n={len(y_win):,}")

    # ── Spread ───────────────────────────────────────────────────────────────
    sp_feats = [f for f in SPREAD_FEATURES if f in df.columns]
    df_sp    = df.dropna(subset=sp_feats + ["home_margin"])
    X_sp     = df_sp[sp_feats].values
    y_sp     = df_sp["home_margin"].values

    if HAS_XGB:
        spread_model, sp_m = _train_xgb_reg(X_sp, y_sp)
    else:
        spread_model, sp_m = _train_ridge(X_sp, y_sp)
    joblib.dump(spread_model, SPREAD_MODEL_PATH)
    metrics["spread_model"] = {**sp_m, "n_samples": int(len(y_sp))}
    logger.info(f"  spread model — rmse={sp_m.get('rmse', '?'):.2f}   n={len(y_sp):,}")

    # ── Total ────────────────────────────────────────────────────────────────
    tot_feats = [f for f in TOTAL_FEATURES if f in df.columns]
    df_tot    = df.dropna(subset=tot_feats + ["total_points"])
    X_tot     = df_tot[tot_feats].values
    y_tot     = df_tot["total_points"].values

    total_model, tot_m = _train_ridge(X_tot, y_tot)
    joblib.dump(total_model, TOTAL_MODEL_PATH)
    metrics["total_model"] = {**tot_m, "n_samples": int(len(y_tot))}
    logger.info(f"  total model  — rmse={tot_m.get('rmse', '?'):.2f}   n={len(y_tot):,}")

    # ── ATS backtest ────────────────────────────────────────────────────────
    metrics["ats"] = _ats_record(df, spread_model, sp_feats)
    logger.info(
        f"  ATS record — {metrics['ats']['wins']}W "
        f"{metrics['ats']['losses']}L  "
        f"({metrics['ats']['pct']:.1%})"
    )

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
    if not all(p.exists() for p in [WIN_MODEL_PATH, SPREAD_MODEL_PATH, TOTAL_MODEL_PATH]):
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
    if not all(p.exists() for p in [WIN_MODEL_PATH, SPREAD_MODEL_PATH, TOTAL_MODEL_PATH]):
        return None
    return _predict_row(row)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add model prediction columns to a feature-matrix DataFrame slice.
    Returns df with new columns: win_prob, predicted_spread, predicted_total.
    """
    if not all(p.exists() for p in [WIN_MODEL_PATH, SPREAD_MODEL_PATH, TOTAL_MODEL_PATH]):
        return df
    win_m    = joblib.load(WIN_MODEL_PATH)
    spread_m = joblib.load(SPREAD_MODEL_PATH)
    total_m  = joblib.load(TOTAL_MODEL_PATH)

    win_feats    = [f for f in WIN_FEATURES    if f in df.columns]
    spread_feats = [f for f in SPREAD_FEATURES if f in df.columns]
    total_feats  = [f for f in TOTAL_FEATURES  if f in df.columns]

    df = df.copy()
    if win_feats:
        X = df[win_feats].fillna(0).values
        df["win_prob"] = _clf_predict_proba(win_m, X)
    if spread_feats:
        X = df[spread_feats].fillna(0).values
        df["predicted_spread"] = _reg_predict(spread_m, X)
    if total_feats:
        X = df[total_feats].fillna(0).values
        df["predicted_total"] = _reg_predict(total_m, X)
    return df


def load_models() -> dict:
    return {
        "win":    joblib.load(WIN_MODEL_PATH)    if WIN_MODEL_PATH.exists()    else None,
        "spread": joblib.load(SPREAD_MODEL_PATH) if SPREAD_MODEL_PATH.exists() else None,
        "total":  joblib.load(TOTAL_MODEL_PATH)  if TOTAL_MODEL_PATH.exists()  else None,
    }


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    return {}


def models_trained() -> bool:
    return all(
        p.exists() for p in [WIN_MODEL_PATH, SPREAD_MODEL_PATH, TOTAL_MODEL_PATH]
    )


# ─────────────────────────────── private helpers ─────────────────────────────

def _get_val(row: pd.Series, col: str, default: float = 0.0) -> float:
    v = row.get(col, default)
    return float(v) if pd.notna(v) else default


def _predict_row(row: pd.Series) -> Prediction:
    win_m    = joblib.load(WIN_MODEL_PATH)
    spread_m = joblib.load(SPREAD_MODEL_PATH)
    total_m  = joblib.load(TOTAL_MODEL_PATH)

    w_x = [[_get_val(row, f) for f in WIN_FEATURES]]
    s_x = [[_get_val(row, f) for f in SPREAD_FEATURES]]
    t_x = [[_get_val(row, f) for f in TOTAL_FEATURES]]

    return Prediction(
        win_prob=float(_clf_predict_proba(win_m, w_x)[0]),
        predicted_spread=float(_reg_predict(spread_m, s_x)[0]),
        predicted_total=float(_reg_predict(total_m, t_x)[0]),
    )


def _clf_predict_proba(model, X):
    if HAS_XGB and isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(X))
    return model.predict_proba(X)[:, 1]


def _reg_predict(model, X):
    if HAS_XGB and isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(X))
    return model.predict(X)


def _tscv(X, y, n_splits: int = 5):
    return list(TimeSeriesSplit(n_splits=n_splits).split(X))


def _train_logreg(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])
    briers = []
    for tr, va in _tscv(X, y):
        pipe.fit(X[tr], y[tr])
        briers.append(brier_score_loss(y[va], pipe.predict_proba(X[va])[:, 1]))
    pipe.fit(X, y)
    return pipe, {"brier": float(np.mean(briers))}


def _train_ridge(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
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
    picked_home_cover = preds > book      # model predicts home covers
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
