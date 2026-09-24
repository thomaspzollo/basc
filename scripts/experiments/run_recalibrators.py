#!/usr/bin/env python3
"""Stage-2 recalibrator ablation (Figure 3(c)).

Replaces BASC's binary isotonic Stage 2 with alternative recalibrators, all fit
on the same single-probe agreement labels:
  iso           binary isotonic, nearest-knot extension (BASC)
  platt         logistic regression on logit(score)
  hist{5,10,20} uniform-mass histogram binning
  platt_bin{5,10}  scaling-binning (Kumar et al., 2019): Platt on one half of
                the calibration set, uniform-mass bins of the Platt outputs on
                the other half
Scores recalibrated (same splits / anchors / probes as the main sweep):
  bascL_k{2,5}  BASC-L's Stage-1 score (fit on D_score); Stage 2 fit on D_iso
  tp            token probability; Stage 2 fit on D_score + D_iso (BASC-TP's data)
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
from sklearn.linear_model import LogisticRegression

from basc import roster
from basc.data import BASCData
from basc.metrics import bundle_all_metrics
from basc.sampling import draw_anchor_set, three_way_split
from basc.semantic import load_anchor_semantic
from basc.stage1 import fit_predict_binomial_logistic
from basc.stage2 import fit_binary_isotonic

HIST_BINS = (5, 10, 20)


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _platt_fit(u_cal, b_cal):
    if len(np.unique(b_cal)) < 2:
        const = float(np.mean(b_cal))
        return lambda u: np.full(len(u), const)
    lr = LogisticRegression(C=1e4, max_iter=1000)
    lr.fit(_logit(u_cal)[:, None], b_cal)
    return lambda u: lr.predict_proba(_logit(u)[:, None])[:, 1]


def platt(u_cal, b_cal, u_test):
    return _platt_fit(u_cal, b_cal)(u_test)


def platt_binning(u_cal, b_cal, u_test, n_bins, rng):
    idx = rng.permutation(len(u_cal))
    t1, t2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
    g = _platt_fit(u_cal[t1], b_cal[t1])
    g2, gt = g(u_cal[t2]), g(u_test)
    edges = np.unique(np.quantile(g2, np.linspace(0, 1, n_bins + 1)[1:-1]))
    bin2 = np.searchsorted(edges, g2, side="right")
    bint = np.searchsorted(edges, gt, side="right")
    means = np.array([g2[bin2 == j].mean() if np.any(bin2 == j) else np.nan for j in range(len(edges) + 1)])
    filled = np.where(np.isfinite(means))[0]          # empty bin -> nearest filled bin
    means = means[filled[np.abs(filled[:, None] - np.arange(len(means))[None, :]).argmin(axis=0)]]
    return means[bint]


def hist_binning(u_cal, b_cal, u_test, n_bins):
    edges = np.unique(np.quantile(u_cal, np.linspace(0, 1, n_bins + 1)[1:-1]))
    cal_bin = np.searchsorted(edges, u_cal, side="right")
    test_bin = np.searchsorted(edges, u_test, side="right")
    means = np.array([b_cal[cal_bin == j].mean() if np.any(cal_bin == j) else np.mean(b_cal)
                      for j in range(len(edges) + 1)])
    return means[test_bin]


def run_combo(job):
    model, dataset, n_seeds, outputs_dir = job
    try:
        data = BASCData.load(dataset, model, outputs_dir=outputs_dir, verbose=False)
    except FileNotFoundError:
        return []
    anchor_sem = load_anchor_semantic(data, outputs_dir=outputs_dir, verbose=False)
    tp_all = np.exp(data.test_avg_logprobs.astype(float)).clip(0, 1)
    max_k = data.k_rollouts
    rows = []
    for seed in range(n_seeds):
        score_pos, iso_pos, test_pos = three_way_split(data.n_queries, seed=seed)
        score = draw_anchor_set(data, score_pos, max_k - 1, seed=seed * 3 + 1, anchor_semantic=anchor_sem)
        iso = draw_anchor_set(data, iso_pos, 1, seed=seed * 3 + 2, anchor_semantic=anchor_sem)
        test = draw_anchor_set(data, test_pos, max_k - 1, seed=seed * 3 + 3, anchor_semantic=anchor_sem)
        agree_test = test.agree[:, 0].astype(np.int64)
        B_iso = iso.agreement[:, 0].astype(np.int64)

        tasks = {}
        for k in (2, 5):
            m = np.full(score.n, k - 1, dtype=np.int64)
            c = score.agreement[:, : k - 1].sum(axis=1).astype(np.int64)
            raw = fit_predict_binomial_logistic(score.X, m, c, np.concatenate([iso.X, test.X]))
            tasks[f"bascL_k{k}"] = (raw[: iso.n], B_iso, raw[iso.n:])
        cal_pos = np.concatenate([score_pos, iso_pos])
        B_cal = np.concatenate([score.agreement[:, 0], iso.agreement[:, 0]]).astype(np.int64)
        tasks["tp"] = (tp_all[cal_pos], B_cal, tp_all[test_pos])

        for score_name, (u_cal, b_cal, u_test) in tasks.items():
            preds = {
                "raw": np.clip(u_test, 0, 1),
                "iso": fit_binary_isotonic(u_cal, b_cal).predict(u_test),
                "platt": platt(u_cal, b_cal, u_test),
            }
            for nb in HIST_BINS:
                preds[f"hist{nb}"] = hist_binning(u_cal, b_cal, u_test, nb)
            for nb in (5, 10):
                preds[f"platt_bin{nb}"] = platt_binning(u_cal, b_cal, u_test, nb,
                                                        np.random.default_rng(seed * 7 + 11))
            for recal, p in preds.items():
                mtr = bundle_all_metrics(p, test.correct, agree_test)
                rows.append({**{kk: v for kk, v in mtr.items() if isinstance(v, (int, float))},
                             "score": score_name, "recal": recal, "model": model, "dataset": dataset,
                             "seed": seed, "n_cal": int(len(u_cal)), "n_distinct": int(len(np.unique(p)))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--outputs_dir", default=None)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    t0 = time.time()
    jobs = [(m, d, args.n_seeds, args.outputs_dir) for m in roster.MODELS for d in roster.datasets_for(m)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(run_combo, jobs) for r in rs]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[write] {args.out_csv}  rows={len(rows)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
