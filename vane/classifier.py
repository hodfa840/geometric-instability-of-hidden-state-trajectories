"""Gradient Boosting classifier, Optuna tuning and nested cross-validation."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .features import VANE_METRICS, build_matrix, labels, sanitize

SEED = 42
N_SPLITS = 5
GB_KEYS = ("n_estimators", "max_depth", "learning_rate", "subsample", "min_samples_leaf")
DEFAULT_GB_PARAMS: Dict[str, float] = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, min_samples_leaf=5
)


def gb_params(params: Optional[dict] = None) -> dict:
    merged = {**DEFAULT_GB_PARAMS, **{k: v for k, v in (params or {}).items() if k in GB_KEYS}}
    merged["random_state"] = SEED
    return merged


def make_folds(y: np.ndarray, n_splits: int = N_SPLITS, seed: int = SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def _fit_predict(model, X_tr, y_tr, X_te) -> np.ndarray:
    scaler = StandardScaler().fit(X_tr)
    model.fit(sanitize(scaler.transform(X_tr)), y_tr)
    return model.predict_proba(sanitize(scaler.transform(X_te)))[:, 1]


@dataclass
class CVResult:
    auroc: float
    auroc_std: float
    fold_aurocs: List[float]
    oof_scores: np.ndarray = field(repr=False)
    fold_params: List[dict] = field(default_factory=list)


def cross_validate(X: np.ndarray, y: np.ndarray, model_fn: Callable[[int], object],
                   folds=None) -> CVResult:
    """K-fold CV with per-fold standardisation; ``model_fn(k)`` builds the model for fold k."""
    X, y = sanitize(X), np.asarray(y)
    if len(np.unique(y)) < 2:
        raise ValueError("labels contain a single class; AUROC is undefined")
    folds = folds if folds is not None else make_folds(y)
    oof = np.zeros(len(y), dtype=np.float64)
    aucs = []
    for k, (tr, te) in enumerate(folds):
        oof[te] = _fit_predict(model_fn(k), X[tr], y[tr], X[te])
        aucs.append(float(roc_auc_score(y[te], oof[te])))
    return CVResult(float(np.mean(aucs)), float(np.std(aucs)), aucs, oof)


def cv_gradient_boosting(X: np.ndarray, y: np.ndarray, params: Optional[dict] = None) -> CVResult:
    p = gb_params(params)
    res = cross_validate(X, y, lambda k: GradientBoostingClassifier(**p))
    res.fold_params = [p] * len(res.fold_aurocs)
    return res


def cv_logistic_regression(X: np.ndarray, y: np.ndarray) -> CVResult:
    return cross_validate(X, y, lambda k: LogisticRegression(max_iter=1000, C=1.0, random_state=SEED))


def tune_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 80, seed: int = SEED) -> dict:
    """Optuna search over Gradient Boosting hyperparameters (5-fold CV AUROC on X, y)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
        )
        return cv_gradient_boosting(X, y, params).auroc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return gb_params(study.best_params)


def nested_cv(X: np.ndarray, y: np.ndarray, n_trials: int = 80, n_jobs: int = 1) -> CVResult:
    """Outer 5-fold CV with hyperparameters tuned on each outer training split only.
    ``n_trials=0`` uses the default hyperparameters."""
    if n_trials <= 0:
        return cv_gradient_boosting(X, y)
    from joblib import Parallel, delayed

    X, y = sanitize(X), np.asarray(y)
    folds = make_folds(y)
    params = Parallel(n_jobs=n_jobs)(
        delayed(tune_hyperparameters)(X[tr], y[tr], n_trials, SEED + k) for k, (tr, _) in enumerate(folds)
    )
    res = cross_validate(X, y, lambda k: GradientBoostingClassifier(**params[k]), folds)
    res.fold_params = list(params)
    return res


def cv_with_fold_params(X: np.ndarray, y: np.ndarray, fold_params: Sequence[dict]) -> CVResult:
    res = cross_validate(X, y, lambda k: GradientBoostingClassifier(**fold_params[k]))
    res.fold_params = list(fold_params)
    return res


class VANEClassifier:
    """StandardScaler + Gradient Boosting on VANE features."""

    def __init__(self, params: Optional[dict] = None, metrics: Sequence[str] = VANE_METRICS):
        self.params = gb_params(params)
        self.metrics = tuple(metrics)
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[GradientBoostingClassifier] = None

    def fit(self, records: Sequence[dict]) -> "VANEClassifier":
        X = build_matrix(records, self.metrics)
        self.scaler = StandardScaler().fit(X)
        self.model = GradientBoostingClassifier(**self.params).fit(sanitize(self.scaler.transform(X)),
                                                                   labels(records))
        return self

    def predict_proba(self, records: Sequence[dict]) -> np.ndarray:
        """P(correct) for each record."""
        if self.model is None:
            raise RuntimeError("classifier is not fitted")
        X = build_matrix(records, self.metrics)
        return self.model.predict_proba(sanitize(self.scaler.transform(X)))[:, 1]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"params": self.params, "metrics": self.metrics,
                         "scaler": self.scaler, "model": self.model}, f)

    @classmethod
    def load(cls, path: str | Path) -> "VANEClassifier":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["params"], d["metrics"])
        obj.scaler, obj.model = d["scaler"], d["model"]
        return obj
