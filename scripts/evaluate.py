#!/usr/bin/env python3
"""Evaluate VANE on an extracted run (CPU): nested-CV AUROC, ablation, log-prob, selective prediction.

    python scripts/evaluate.py results/llama-3-8b-instruct_gsm8k --n_jobs 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vane.classifier import (VANEClassifier, cv_logistic_regression, cv_with_fold_params, gb_params,  # noqa: E402
                             nested_cv, tune_hyperparameters)
from vane.evaluation import load_records, score_auroc, selective_accuracy  # noqa: E402
from vane.features import VANE_METRICS, build_matrix, labels  # noqa: E402


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--optuna_trials", type=int, default=80, help="0: default hyperparameters, no tuning")
    p.add_argument("--n_jobs", type=int, default=1, help="parallel outer folds")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    records = load_records(run_dir)
    y = labels(records)
    X = build_matrix(records)
    print(f"{run_dir.name}: N={len(y)}  accuracy={y.mean():.4f}  D={X.shape[1]}", flush=True)

    vane = nested_cv(X, y, n_trials=args.optuna_trials, n_jobs=args.n_jobs)
    print(f"VANE      AUROC {pct(vane.auroc)} +- {pct(vane.auroc_std)}", flush=True)
    out = {"n": int(len(y)), "accuracy": float(y.mean()), "feature_dim": int(X.shape[1]),
           "optuna_trials": args.optuna_trials,
           "vane": {"auroc": vane.auroc, "auroc_std": vane.auroc_std, "fold_aurocs": vane.fold_aurocs,
                    "fold_params": vane.fold_params}}

    abl = {}
    for m in VANE_METRICS:
        r = cv_logistic_regression(build_matrix(records, (m,)), y)
        abl[m] = {"auroc": r.auroc, "auroc_std": r.auroc_std}
    r = cv_logistic_regression(X, y)
    abl["all_geometric_lr"] = {"auroc": r.auroc, "auroc_std": r.auroc_std}
    out["ablation"] = abl
    print("ablation  " + "  ".join(f"{k}={pct(v['auroc'])}" for k, v in abl.items()))

    lp = np.array([float(r["mean_log_prob"]) for r in records])
    out["log_prob"] = score_auroc(lp, y)
    print(f"Log-Prob  AUROC {pct(out['log_prob']['auroc'])} +- {pct(out['log_prob']['auroc_std'])}")

    both = cv_with_fold_params(np.hstack([X, lp[:, None]]), y, vane.fold_params)
    r_lp = float(np.corrcoef(vane.oof_scores, lp)[0, 1])
    out["orthogonality"] = {"vane_plus_logprob_auroc": both.auroc, "delta": both.auroc - vane.auroc,
                            "pearson_r_vane_vs_logprob": r_lp}
    print(f"VANE+LP   AUROC {pct(both.auroc)} (delta {100 * (both.auroc - vane.auroc):+.1f})  r={r_lp:.2f}")

    out["selective_prediction"] = {"vane": selective_accuracy(vane.oof_scores, y),
                                   "log_prob": selective_accuracy(lp, y)}
    sp = out["selective_prediction"]
    print("coverage  " + "  ".join(sp["vane"]))
    print("VANE acc  " + "  ".join(f"{v:.3f}" for v in sp["vane"].values()))
    print("LP acc    " + "  ".join(f"{v:.3f}" for v in sp["log_prob"].values()))
    np.save(run_dir / "vane_oof_scores.npy", vane.oof_scores)

    params = tune_hyperparameters(X, y, args.optuna_trials) if args.optuna_trials > 0 else gb_params()
    VANEClassifier(params).fit(records).save(run_dir / "classifier.pkl")
    out["deployed_params"] = params
    (run_dir / "eval.json").write_text(json.dumps(out, indent=2))
    print(f"-> {run_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
