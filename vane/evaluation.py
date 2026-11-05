"""Log-probability baseline, selective prediction and run I/O."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .classifier import cv_logistic_regression

COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def score_auroc(score: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """5-fold CV AUROC of a scalar score (1-D logistic regression)."""
    res = cv_logistic_regression(np.asarray(score, dtype=np.float64).reshape(-1, 1), y)
    return {"auroc": res.auroc, "auroc_std": res.auroc_std}


def selective_accuracy(scores: np.ndarray, y: np.ndarray, coverages: Sequence[float] = COVERAGES) -> Dict[str, float]:
    """Accuracy on the highest-scoring fraction of samples at each coverage."""
    scores, y = np.asarray(scores), np.asarray(y)
    order = np.argsort(scores, kind="stable")
    return {f"{c:.2f}": float(y[order[-max(1, int(c * len(y))):]].mean()) for c in coverages}


def load_records(run_dir: str | Path) -> List[dict]:
    with open(Path(run_dir) / "records.pkl", "rb") as f:
        return sorted(pickle.load(f), key=lambda r: r["sample_idx"])
