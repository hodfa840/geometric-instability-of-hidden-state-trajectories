"""End-to-end checks of features -> nested CV -> classifier -> selective prediction."""
import numpy as np
import torch

from vane.classifier import VANEClassifier, cv_gradient_boosting, nested_cv
from vane.evaluation import score_auroc, selective_accuracy
from vane.features import build_matrix, labels
from vane.metrics import compute_profiles


def synthetic_records(n=120, L=8, T=10, d=6, seed=0, signal=True):
    """With ``signal``, incorrect samples get noisier (more unstable) trajectories."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        y = int(rng.random() < 0.5)
        scale = (0.2 if y else 1.0) if signal else 0.6
        steps = torch.from_numpy(rng.normal(size=(L, T, d))).float() * scale
        rec = compute_profiles(torch.cumsum(steps + 1.0, dim=0))
        rec.update(is_correct=y, sample_idx=i, mean_log_prob=float(rng.normal()))
        recs.append(rec)
    return recs


def test_end_to_end():
    recs = synthetic_records()
    X, y = build_matrix(recs), labels(recs)
    res = cv_gradient_boosting(X, y, {"n_estimators": 50})
    assert res.auroc > 0.9 and len(res.fold_aurocs) == 5 and res.oof_scores.shape == (len(y),)
    sel = selective_accuracy(res.oof_scores, y)
    assert sel["0.50"] >= sel["1.00"]
    clf = VANEClassifier({"n_estimators": 50}).fit(recs)
    assert clf.predict_proba(recs[:3]).shape == (3,)
    lp = score_auroc(np.array([r["mean_log_prob"] for r in recs]), y)
    assert 0.3 < lp["auroc"] < 0.7                       # random score


def test_nested_cv_tunes_per_fold():
    recs = synthetic_records(n=80)
    res = nested_cv(build_matrix(recs), labels(recs), n_trials=2)
    assert len(res.fold_params) == 5 and res.auroc > 0.8


def test_nested_cv_no_leakage_on_random_labels():
    # Selecting hyperparameters on the test folds would inflate AUROC on pure noise;
    # nested CV must stay at chance level.
    recs = synthetic_records(n=150, signal=False, seed=1)
    res = nested_cv(build_matrix(recs), labels(recs), n_trials=4)
    assert abs(res.auroc - 0.5) < 0.12
