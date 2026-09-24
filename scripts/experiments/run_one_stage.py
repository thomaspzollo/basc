#!/usr/bin/env python3
""""1-stage" ablation (Figure 3(a)): BASC-L's binomial-logistic Stage 1 trained on
all 400 calibration questions (D_score + D_iso) with no Stage-2 recalibration.

Method key: onestage_binomial_k{k}.
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
from basc.data import BASCData
from basc.metrics import bundle_all_metrics
from basc.sampling import draw_anchor_set, three_way_split
from basc.semantic import load_anchor_semantic
from basc.stage1 import fit_predict_binomial_logistic


def run_combo(job):
    model, dataset, ks, n_seeds, outputs_dir = job
    try:
        data = BASCData.load(dataset, model, outputs_dir=outputs_dir, verbose=False)
    except FileNotFoundError:
        return []
    anchor_sem = load_anchor_semantic(data, outputs_dir=outputs_dir, verbose=False)
    max_k = data.k_rollouts
    rows = []
    for seed in range(n_seeds):
        score_pos, iso_pos, test_pos = three_way_split(data.n_queries, seed=seed)
        score = draw_anchor_set(data, score_pos, max_k - 1, seed=seed * 3 + 1, anchor_semantic=anchor_sem)
        iso = draw_anchor_set(data, iso_pos, max_k - 1, seed=seed * 3 + 2, anchor_semantic=anchor_sem)
        test = draw_anchor_set(data, test_pos, 1, seed=seed * 3 + 3, anchor_semantic=anchor_sem)
        X_train = np.concatenate([score.X, iso.X], axis=0)
        agree_train = np.concatenate([score.agreement, iso.agreement], axis=0)
        agree_test = test.agreement[:, 0].astype(np.int64)
        for k in ks:
            if k < 2 or k > max_k:
                continue
            m = np.full(X_train.shape[0], k - 1, dtype=np.int64)
            c = agree_train[:, : k - 1].sum(axis=1).astype(np.int64)
            preds = fit_predict_binomial_logistic(X_train, m, c, test.X)
            metrics = bundle_all_metrics(preds, test.correct, agree_test)
            rows.append({
                **{kk: v for kk, v in metrics.items() if isinstance(v, (int, float))},
                "method": f"onestage_binomial_k{k}", "k": k, "model": model, "dataset": dataset,
                "seed": seed, "accuracy": float(test.correct.mean()), "n_test": int(test.n),
                "n_train_queries": int(X_train.shape[0]),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 5, 10, 20, 100])
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--outputs_dir", default=None)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    t0 = time.time()
    jobs = [(m, d, tuple(args.ks), args.n_seeds, args.outputs_dir)
            for m in roster.MODELS for d in roster.datasets_for(m)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(run_combo, jobs) for r in rs]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[write] {args.out_csv}  rows={len(rows)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
