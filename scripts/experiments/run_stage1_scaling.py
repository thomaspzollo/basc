#!/usr/bin/env python3
"""Stage-1 estimator ablation (Figure 3(b)).

With BASC's binary isotonic Stage 2 held fixed, compare two Stage-1 score
models trained on D_score with k-1 probes per question:
  binomial  BASC-L's binomial-logistic regression on the binary agreements
  ridge     ridge regression on the empirical self-consistency rate (RR-SC)
for k in {2, 3, 5, 10, 20, 50, 100}. Rows are recorded both before
(stage2="raw") and after (stage2="iso_binary") Stage 2.
"""
from __future__ import annotations

import os

# The paper's results were computed with 16 BLAS threads. The Stage-1 lbfgs
# fits are sensitive to the BLAS thread count (see README, "Reproducibility"),
# so it is pinned before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "16")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from basc import roster
from basc.baselines import fit_predict_ridge_clip
from basc.data import BASCData
from basc.metrics import bundle_all_metrics
from basc.sampling import draw_anchor_set, empirical_sc, three_way_split
from basc.semantic import load_anchor_semantic
from basc.stage1 import fit_predict_binomial_logistic
from basc.stage2 import fit_binary_isotonic

K_VALUES = (2, 3, 5, 10, 20, 50, 100)


def run_trial(data, seed, anchor_sem):
    max_k = data.k_rollouts
    score_pos, iso_pos, test_pos = three_way_split(data.n_queries, seed=seed)
    score = draw_anchor_set(data, score_pos, max_k - 1, seed=seed * 3 + 1, anchor_semantic=anchor_sem)
    iso = draw_anchor_set(data, iso_pos, 1, seed=seed * 3 + 2, anchor_semantic=anchor_sem)
    test = draw_anchor_set(data, test_pos, 1, seed=seed * 3 + 3, anchor_semantic=anchor_sem)
    agree_score = score.agreement
    B_iso = iso.agreement[:, 0].astype(np.int64)
    X_iso_test = np.concatenate([iso.X, test.X], axis=0)

    results = {}
    for k in K_VALUES:
        if k > max_k:
            continue
        y_sc = empirical_sc(agree_score[:, : k - 1])
        U = fit_predict_ridge_clip(score.X, y_sc, X_iso_test, random_state=seed)
        results[("ridge", "raw", k)] = np.clip(U[iso.n:], 0.0, 1.0)
        results[("ridge", "iso_binary", k)] = fit_binary_isotonic(U[: iso.n], B_iso).predict(U[iso.n:])

        m = np.full(score.n, k - 1, dtype=np.int64)
        c = agree_score[:, : k - 1].sum(axis=1).astype(np.int64)
        U = fit_predict_binomial_logistic(score.X, m, c, X_iso_test)
        results[("binomial", "raw", k)] = np.clip(U[iso.n:], 0.0, 1.0)
        results[("binomial", "iso_binary", k)] = fit_binary_isotonic(U[: iso.n], B_iso).predict(U[iso.n:])
    return results, test.correct, test.agreement[:, 0].astype(np.int64)


def run_combo(job):
    model, dataset, n_seeds, outputs_dir = job
    try:
        data = BASCData.load(dataset, model, outputs_dir=outputs_dir, verbose=False)
    except FileNotFoundError:
        return []
    anchor_sem = load_anchor_semantic(data, outputs_dir=outputs_dir, verbose=False)
    rows = []
    for seed in range(n_seeds):
        results, correct, agree = run_trial(data, seed, anchor_sem)
        for (target, stage2, k), preds in results.items():
            m = bundle_all_metrics(preds, correct, agree)
            rows.append({**{kk: v for kk, v in m.items() if isinstance(v, (int, float))},
                         "target": target, "stage2": stage2, "k": k,
                         "model": model, "dataset": dataset, "seed": seed})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--outputs_dir", default=None)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    t0 = time.time()
    jobs = [(m, d, args.n_seeds, args.outputs_dir) for m, d in roster.combos()]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(run_combo, jobs) for r in rs]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[write] {args.out_csv}  rows={len(rows)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
