"""Market-anchored challenger models with nested temporal shrinkage."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.temporal import walk_forward_season_splits


def _fallback_regression(model, frame: pd.DataFrame) -> np.ndarray:
    """Predict with either a sklearn estimator or a native XGBoost booster."""
    try:
        import xgboost as xgb
        if isinstance(model, xgb.Booster):
            matrix = xgb.DMatrix(
                frame.to_numpy(), feature_names=[str(column) for column in frame.columns]
            )
            return np.asarray(model.predict(matrix), dtype=float)
    except ImportError:
        pass
    return np.asarray(model.predict(frame), dtype=float)


def _fallback_probability(model, frame: pd.DataFrame) -> np.ndarray:
    """Return P(class=1) for either sklearn or native XGBoost models."""
    try:
        import xgboost as xgb
        if isinstance(model, xgb.Booster):
            matrix = xgb.DMatrix(
                frame.to_numpy(), feature_names=[str(column) for column in frame.columns]
            )
            return np.asarray(model.predict(matrix), dtype=float)
    except ImportError:
        pass
    return np.asarray(model.predict_proba(frame)[:, 1], dtype=float)


def _regressor(kind: str):
    if kind == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=100.0)),
            ]
        )
    if kind == "hist":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.04,
                        max_iter=250,
                        max_leaf_nodes=15,
                        l2_regularization=10.0,
                        min_samples_leaf=40,
                        random_state=42,
                    ),
                ),
            ]
        )
    if kind == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=350,
                        min_samples_leaf=20,
                        max_features=0.7,
                        n_jobs=4,
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown residual model: {kind}")


def _optimal_shrinkage(prediction: np.ndarray, target: np.ndarray, cap: float = 1.0) -> float:
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[valid], target[valid]
    denominator = float(np.dot(prediction, prediction))
    if len(prediction) < 100 or denominator <= 1e-12:
        return 0.0
    coefficient = float(np.dot(prediction, target) / denominator)
    return float(np.clip(coefficient, 0.0, cap))


def _nested_residual_shrinkage(
    X: pd.DataFrame,
    residual: np.ndarray,
    temporal: pd.DataFrame,
    *,
    kind: str,
) -> tuple[float, int]:
    predictions = np.full(len(residual), np.nan)
    inner_folds = list(walk_forward_season_splits(temporal, min_train_seasons=1))
    for fold in inner_folds:
        train = fold.train_index[np.isfinite(residual[fold.train_index])]
        test = fold.test_index[np.isfinite(residual[fold.test_index])]
        if len(train) < 200 or not len(test):
            continue
        model = _regressor(kind)
        model.fit(X.iloc[train], residual[train])
        predictions[test] = model.predict(X.iloc[test])
    valid = np.isfinite(predictions) & np.isfinite(residual)
    return _optimal_shrinkage(predictions[valid], residual[valid]), int(valid.sum())


def walk_forward_market_residual(
    X: pd.DataFrame,
    target: np.ndarray,
    market_prediction: np.ndarray,
    temporal: pd.DataFrame,
    *,
    kind: str = "ridge",
    min_train_seasons: int = 2,
) -> tuple[np.ndarray, list[dict]]:
    """Predict market errors, shrinking corrections using inner OOS seasons."""
    y = np.asarray(target, dtype=float)
    market = np.asarray(market_prediction, dtype=float)
    residual = y - market
    oof = np.full(len(y), np.nan)
    metadata: list[dict] = []
    for fold in walk_forward_season_splits(
        temporal, min_train_seasons=min_train_seasons
    ):
        train = fold.train_index[
            np.isfinite(residual[fold.train_index])
            & np.isfinite(market[fold.train_index])
        ]
        test = fold.test_index[
            np.isfinite(y[fold.test_index]) & np.isfinite(market[fold.test_index])
        ]
        if len(train) < 300 or not len(test):
            continue
        shrinkage, shrinkage_n = _nested_residual_shrinkage(
            X.iloc[train].reset_index(drop=True),
            residual[train],
            temporal.iloc[train].reset_index(drop=True),
            kind=kind,
        )
        model = _regressor(kind)
        model.fit(X.iloc[train], residual[train])
        correction = model.predict(X.iloc[test])
        oof[test] = market[test] + shrinkage * correction
        metadata.append(
            {
                "fold": fold.fold,
                "test_season": fold.test_season,
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "shrinkage": shrinkage,
                "shrinkage_oos_n": shrinkage_n,
            }
        )
    return oof, metadata


def fit_market_residual(
    X: pd.DataFrame,
    target: np.ndarray,
    market_prediction: np.ndarray,
    temporal: pd.DataFrame,
    *,
    kind: str,
):
    y = np.asarray(target, dtype=float)
    market = np.asarray(market_prediction, dtype=float)
    residual = y - market
    valid = np.isfinite(residual) & np.isfinite(market)
    shrinkage, shrinkage_n = _nested_residual_shrinkage(
        X.loc[valid].reset_index(drop=True),
        residual[valid],
        temporal.loc[valid].reset_index(drop=True),
        kind=kind,
    )
    model = _regressor(kind)
    model.fit(X.loc[valid], residual[valid])
    return model, shrinkage, shrinkage_n


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def walk_forward_market_calibration(
    market_probability: np.ndarray,
    target: np.ndarray,
    temporal: pd.DataFrame,
    *,
    min_train_seasons: int = 2,
) -> tuple[np.ndarray, list[dict]]:
    """Fit a low-variance favourite/longshot correction to no-vig prices."""
    market = np.asarray(market_probability, dtype=float)
    y = np.asarray(target, dtype=float)
    oof = np.full(len(y), np.nan)
    metadata: list[dict] = []
    for fold in walk_forward_season_splits(
        temporal, min_train_seasons=min_train_seasons
    ):
        train = fold.train_index[
            np.isfinite(market[fold.train_index]) & np.isfinite(y[fold.train_index])
        ]
        test = fold.test_index[
            np.isfinite(market[fold.test_index]) & np.isfinite(y[fold.test_index])
        ]
        if len(train) < 200 or not len(test):
            continue
        model = LogisticRegression(C=0.1, max_iter=2000)
        model.fit(logit(market[train]).reshape(-1, 1), y[train].astype(int))
        oof[test] = model.predict_proba(logit(market[test]).reshape(-1, 1))[:, 1]
        metadata.append(
            {
                "fold": fold.fold,
                "test_season": fold.test_season,
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "intercept": float(model.intercept_[0]),
                "slope": float(model.coef_[0, 0]),
            }
        )
    return oof, metadata


def walk_forward_market_structural(
    X: pd.DataFrame,
    market_probability: np.ndarray,
    target: np.ndarray,
    temporal: pd.DataFrame,
    *,
    c: float = 0.05,
    min_train_seasons: int = 2,
) -> tuple[np.ndarray, list[dict]]:
    """Combine the market logit with point-in-time structural signals."""
    market = np.asarray(market_probability, dtype=float)
    y = np.asarray(target, dtype=float)
    design = X.copy()
    design.insert(0, "market_logit", logit(market))
    design.loc[~np.isfinite(market), "market_logit"] = np.nan
    oof = np.full(len(y), np.nan)
    metadata: list[dict] = []
    for fold in walk_forward_season_splits(
        temporal, min_train_seasons=min_train_seasons
    ):
        train = fold.train_index[
            np.isfinite(market[fold.train_index]) & np.isfinite(y[fold.train_index])
        ]
        test = fold.test_index[
            np.isfinite(market[fold.test_index]) & np.isfinite(y[fold.test_index])
        ]
        if len(train) < 200 or not len(test):
            continue
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=c, max_iter=2000)),
            ]
        )
        model.fit(design.iloc[train], y[train].astype(int))
        oof[test] = model.predict_proba(design.iloc[test])[:, 1]
        metadata.append(
            {
                "fold": fold.fold,
                "test_season": fold.test_season,
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "c": c,
            }
        )
    return oof, metadata


def fit_market_calibrator(market_probability: np.ndarray, target: np.ndarray):
    market = np.asarray(market_probability, dtype=float)
    y = np.asarray(target, dtype=float)
    valid = np.isfinite(market) & np.isfinite(y)
    model = LogisticRegression(C=0.1, max_iter=2000)
    model.fit(logit(market[valid]).reshape(-1, 1), y[valid].astype(int))
    return model


@dataclass
class MarketAnchoredRegressor:
    residual_model: object
    fallback_model: object
    feature_names: tuple[str, ...]
    market_feature: str
    market_sign: float
    shrinkage: float

    def predict(self, X) -> np.ndarray:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        fallback = _fallback_regression(self.fallback_model, frame)
        market_raw = pd.to_numeric(frame[self.market_feature], errors="coerce").to_numpy(float)
        valid = np.isfinite(market_raw)
        if valid.any():
            correction = np.asarray(self.residual_model.predict(frame.loc[valid]), dtype=float)
            fallback[valid] = self.market_sign * market_raw[valid] + self.shrinkage * correction
        return fallback


@dataclass
class MarketBaselineRegressor:
    """Use the observed market consensus, falling back to a structural model."""
    fallback_model: object
    feature_names: tuple[str, ...]
    market_feature: str
    market_sign: float

    def predict(self, X) -> np.ndarray:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        prediction = _fallback_regression(self.fallback_model, frame)
        market = pd.to_numeric(frame[self.market_feature], errors="coerce").to_numpy(float)
        valid = np.isfinite(market)
        prediction[valid] = self.market_sign * market[valid]
        return prediction


@dataclass
class MarketBaselineClassifier:
    """Use no-vig market probability, falling back to a structural classifier."""
    fallback_model: object
    feature_names: tuple[str, ...]
    market_feature: str = "market_home_prob"

    def predict_proba(self, X) -> np.ndarray:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        home_probability = _fallback_probability(self.fallback_model, frame)
        market = pd.to_numeric(frame[self.market_feature], errors="coerce").to_numpy(float)
        valid = np.isfinite(market)
        home_probability[valid] = market[valid]
        home_probability = np.clip(home_probability, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - home_probability, home_probability])


@dataclass
class MarketCalibratedClassifier:
    market_model: object
    fallback_model: object
    feature_names: tuple[str, ...]
    market_feature: str = "market_home_prob"

    def predict_proba(self, X) -> np.ndarray:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        fallback_home = _fallback_probability(self.fallback_model, frame)
        fallback = np.column_stack([1 - fallback_home, fallback_home])
        market = pd.to_numeric(frame[self.market_feature], errors="coerce").to_numpy(float)
        valid = np.isfinite(market)
        if valid.any():
            calibrated = self.market_model.predict_proba(logit(market[valid]).reshape(-1, 1))
            fallback[valid] = calibrated
        return fallback
