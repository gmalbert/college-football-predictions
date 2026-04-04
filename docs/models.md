# Suggested Models Roadmap

**Status:** ✅ Completed — implemented in `utils/models.py` with XGBoost, Ridge, and Elo modeling.

This document outlines the modeling strategy — from simple baselines to
production-grade ensembles — with concrete code scaffolding you can build on.

---

## 1. Modeling Goals

**Status:** ✅ Completed

| Goal | Target Variable | Type |
|------|----------------|------|
| **Win Probability** | Home team win (1/0) | Binary classification |
| **Point Spread** | Home score − Away score | Regression |
| **Total Points (O/U)** | Home score + Away score | Regression |
| **Value Bet Detection** | Model line − Book line (edge) | Derived |

---

## 2. Baseline Models (Ship First)

**Status:** ✅ Completed

### 2a. Elo Rating Model

**Status:** ✅ Completed

A simple, interpretable model that updates after every game. Great starting
point with no feature engineering required.

```python
"""elo.py — Elo rating system for college football."""
from __future__ import annotations

K_FACTOR = 20          # sensitivity to individual results
HOME_ADVANTAGE = 65    # Elo points added for home team
INITIAL_ELO = 1500
REVERSION_FACTOR = 0.33  # mean-revert 1/3 toward 1500 each off-season

def expected_win_prob(elo_a: float, elo_b: float) -> float:
    """Return the expected probability that team A beats team B."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(
    winner_elo: float,
    loser_elo: float,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """Return (new_winner_elo, new_loser_elo) after a game."""
    exp_w = expected_win_prob(winner_elo, loser_elo)
    new_winner = winner_elo + k * (1 - exp_w)
    new_loser = loser_elo + k * (0 - (1 - exp_w))
    return new_winner, new_loser

def season_revert(elo: float) -> float:
    """Mean-revert Elo ratings between seasons."""
    return elo + REVERSION_FACTOR * (INITIAL_ELO - elo)


class EloModel:
    """Track Elo ratings across a full season."""

    def __init__(self, k: float = K_FACTOR, home_adv: float = HOME_ADVANTAGE):
        self.k = k
        self.home_adv = home_adv
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.setdefault(team, INITIAL_ELO)

    def predict(self, home: str, away: str) -> float:
        """Return P(home win)."""
        home_elo = self.get_rating(home) + self.home_adv
        away_elo = self.get_rating(away)
        return expected_win_prob(home_elo, away_elo)

    def update(self, home: str, away: str, home_won: bool) -> None:
        home_elo = self.get_rating(home) + self.home_adv
        away_elo = self.get_rating(away)
        if home_won:
            new_h, new_a = update_elo(home_elo, away_elo, self.k)
        else:
            new_a, new_h = update_elo(away_elo, home_elo, self.k)
        self.ratings[home] = new_h - self.home_adv
        self.ratings[away] = new_a

    def new_season(self) -> None:
        for team in self.ratings:
            self.ratings[team] = season_revert(self.ratings[team])
```

### 2b. Logistic Regression (Spread / Win Prob)

**Status:** ✅ Completed

Fast to train, easy to explain, strong baseline.

```python
"""logistic_baseline.py — Logistic regression win-probability model."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    "elo_diff",
    "sp_plus_diff",
    "off_epa_diff",
    "def_epa_diff",
    "turnover_margin_diff",
    "recruiting_diff",
    "home_flag",
]

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])

def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Walk-forward cross-validation using TimeSeriesSplit."""
    X = df[FEATURE_COLS].values
    y = df["home_win"].values
    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    for train_idx, test_idx in tscv.split(X):
        pipe = build_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        probs = pipe.predict_proba(X[test_idx])[:, 1]
        results.append({
            "brier": brier_score_loss(y[test_idx], probs),
            "log_loss": log_loss(y[test_idx], probs),
        })
    return {
        "mean_brier": sum(r["brier"] for r in results) / len(results),
        "mean_log_loss": sum(r["log_loss"] for r in results) / len(results),
    }
```

---

## 3. Intermediate Models

**Status:** ✅ Completed

### 3a. Gradient-Boosted Trees (XGBoost / LightGBM)

**Status:** ✅ Completed

The workhorse of tabular prediction tasks. Handles non-linearities and
feature interactions out of the box.

```python
"""xgb_model.py — XGBoost spread & win-probability model."""
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def train_xgb_spread(df: pd.DataFrame, feature_cols: list[str]) -> xgb.Booster:
    """Train an XGBoost model to predict the point spread."""
    X = df[feature_cols]
    y = df["home_margin"]  # home_score - away_score

    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_rmse = float("inf")

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "verbosity": 0,
    }

    for train_idx, val_idx in tscv.split(X):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
        dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        preds = model.predict(dval)
        rmse = np.sqrt(np.mean((preds - y.iloc[val_idx].values) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    return best_model


def train_xgb_win_prob(df: pd.DataFrame, feature_cols: list[str]) -> xgb.Booster:
    """Train an XGBoost model for win probability (binary classification)."""
    X = df[feature_cols]
    y = df["home_win"].astype(int)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "verbosity": 0,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_ll = float("inf")

    for train_idx, val_idx in tscv.split(X):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
        dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        preds = model.predict(dval)
        ll = -np.mean(
            y.iloc[val_idx].values * np.log(preds + 1e-10)
            + (1 - y.iloc[val_idx].values) * np.log(1 - preds + 1e-10)
        )
        if ll < best_ll:
            best_ll = ll
            best_model = model

    return best_model
```

### 3b. Ridge Regression for Totals (O/U)

**Status:** ✅ Completed

```python
"""ridge_total.py — Ridge regression for over/under prediction."""
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TOTAL_FEATURES = [
    "home_off_epa",
    "away_off_epa",
    "home_def_epa",
    "away_def_epa",
    "home_pace",
    "away_pace",
    "home_off_success_rate",
    "away_off_success_rate",
    "weather_wind",
    "weather_temp",
    "is_dome",
]

def build_total_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
```

---

## 4. Advanced Models (Phase 2+)

### 4a. Stacking Ensemble

Combine multiple base models into a meta-learner for improved accuracy.

```python
"""ensemble.py — Stacking ensemble combining base models."""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

class StackingEnsemble:
    """
    Level-0: individual model predictions (Elo, logreg, XGBoost)
    Level-1: meta-learner (logistic regression for classification,
             ridge for regression)
    """

    def __init__(self, task: str = "classification"):
        self.task = task
        if task == "classification":
            self.meta = LogisticRegression(C=1.0, max_iter=500)
        else:
            self.meta = Ridge(alpha=1.0)

    def fit(self, base_preds: np.ndarray, y: np.ndarray) -> None:
        """
        base_preds: (n_samples, n_models) array of level-0 predictions.
        y: true labels.
        """
        self.meta.fit(base_preds, y)

    def predict(self, base_preds: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return self.meta.predict_proba(base_preds)[:, 1]
        return self.meta.predict(base_preds)
```

### 4b. Deep Learning (TBD)

- **LSTM / Transformer** on play-by-play sequences for live win-probability.
- **Graph Neural Network** on the schedule graph to capture strength-of-
  schedule relationships.
- Only pursue once the tabular models plateau; deep learning rarely beats
  GBM on structured data without massive feature engineering.

---

## 5. Model Evaluation Framework

**Status:** ✅ Completed

| Metric | Use Case | Target |
|--------|----------|--------|
| **Brier Score** | Win probability calibration | < 0.20 |
| **Log Loss** | Win probability sharpness | < 0.55 |
| **RMSE (spread)** | Spread prediction accuracy | < 14 pts |
| **RMSE (total)** | O/U prediction accuracy | < 12 pts |
| **ATS Record** | Betting accuracy vs. spread | > 52.4% (break-even at −110) |
| **ROI** | Return on flat $100 bets | > 0% |
| **Calibration Curve** | Predicted vs actual probability | Hugging the diagonal |

```python
"""evaluation.py — Model evaluation utilities."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

def evaluate_win_model(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "accuracy": np.mean((y_prob > 0.5) == y_true),
    }

def evaluate_spread_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_true - y_pred
    return {
        "rmse": np.sqrt(np.mean(residuals ** 2)),
        "mae": np.mean(np.abs(residuals)),
        "bias": np.mean(residuals),
    }

def ats_record(
    predicted_spread: np.ndarray,
    book_spread: np.ndarray,
    actual_margin: np.ndarray,
) -> dict:
    """Evaluate against-the-spread betting performance."""
    pick_home = predicted_spread < book_spread  # model says home covers
    home_covered = actual_margin > -book_spread
    pushes = actual_margin == -book_spread

    wins = np.sum(pick_home == home_covered) - np.sum(pushes)
    losses = np.sum(pick_home != home_covered) - np.sum(pushes)
    total = wins + losses
    return {
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(np.sum(pushes)),
        "win_pct": wins / total if total > 0 else 0.0,
    }
```

---

## 6. Model Training Schedule

**Status:** ✅ Completed

| Phase | When | What |
|-------|------|------|
| **Off-season retrain** | August 1 | Full retrain on last 10 years of data |
| **Weekly update** | Monday morning | Incremental update with latest week's results |
| **Pre-game refresh** | 6 hrs before kickoff | Re-pull lines; re-run predictions |
| **Live (future)** | During games | Update win-prob from play-by-play |

---

## 7. Model Persistence

**Status:** ✅ Completed

```python
import joblib
from pathlib import Path

MODEL_DIR = Path("data_files/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def save_model(model, name: str) -> Path:
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path

def load_model(name: str):
    path = MODEL_DIR / f"{name}.joblib"
    return joblib.load(path)
```
