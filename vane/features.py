"""VANE feature vector (Section 3.3, Algorithm 1)."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from .metrics import AGGREGATIONS, METRIC_PREFIXES

VANE_METRICS: Tuple[str, ...] = tuple(METRIC_PREFIXES)


def build_features(record: dict, metrics: Sequence[str] = VANE_METRICS) -> np.ndarray:
    """Concatenate the per-layer profiles of all metrics and aggregations."""
    parts = []
    for metric in metrics:
        if metric not in METRIC_PREFIXES:
            raise ValueError(f"unknown metric '{metric}'; choose from {VANE_METRICS}")
        prefix = METRIC_PREFIXES[metric]
        for agg in AGGREGATIONS:
            parts.append(np.asarray(record[f"{prefix}_{agg}"], dtype=np.float64).ravel())
    return np.concatenate(parts)


def build_matrix(records: Sequence[dict], metrics: Sequence[str] = VANE_METRICS) -> np.ndarray:
    return sanitize(np.stack([build_features(r, metrics) for r in records], axis=0))


def labels(records: Iterable[dict]) -> np.ndarray:
    return np.array([int(r["is_correct"]) for r in records], dtype=np.int64)


def sanitize(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
