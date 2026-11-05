"""VANE: Geometric Instability of Hidden-State Trajectories Predicts Reasoning Failures in LLMs."""
from .classifier import VANEClassifier, nested_cv, tune_hyperparameters
from .features import VANE_METRICS, build_features, build_matrix
from .metrics import compute_profiles, per_token_metrics

__all__ = [
    "compute_profiles",
    "per_token_metrics",
    "VANE_METRICS",
    "build_features",
    "build_matrix",
    "VANEClassifier",
    "nested_cv",
    "tune_hyperparameters",
]
__version__ = "1.0.0"
