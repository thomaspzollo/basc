#!/usr/bin/env python3
"""BASC-TP: BASC's Stage 2 applied directly to token probabilities (no Stage 1).

The score is the deployed response's token probability (TP; ATP is also
recorded). Stage 2 is fit on (score, single-probe agreement) using exactly the
anchors and probes of the main / shift sweeps:
  {tp,atp}_raw        the uncalibrated score
  {tp,atp}_iso_diso   Stage 2 on D_iso only (200 questions, 400 decodes)
  {tp,atp}_iso_dcal   Stage 2 on D_score + D_iso with each question's first
                      probe (400 questions, 800 decodes) -- "BASC-TP" in the paper

--setting in_distribution  -> Table 1 / Figure 3(a)   (results/basc_tp.csv)
--setting shift            -> Table 2                 (results/basc_tp_shift.csv)
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
from basc.experiment import positions_for_ids, split_train_for_shift
from basc.metrics import bundle_all_metrics, correctness_metrics
from basc.sampling import draw_anchor_set, three_way_split
from basc.semantic import load_anchor_semantic
from basc.shift import LANGUAGE_SPLITS, MATH_PAIRS, QA_PAIRS, lang_ids, lang_label
from basc.stage2 import fit_binary_isotonic


def _token_prob_scores(data):
    return {"tp": np.exp(data.test_avg_logprobs.astype(float)),
            "atp": np.exp(data.test_ans_logprobs.astype(float))}


def _fit_variants(name, s_cal_src, s_eval, iso_pos, cal_pos, eval_pos, B_iso, B_cal):
    fill = float(np.nanmedian(s_cal_src[cal_pos]))       # guard only; no NaNs expected
    s_cal_src = np.where(np.isfinite(s_cal_src), s_cal_src, fill).clip(0.0, 1.0)
    s_eval = np.where(np.isfinite(s_eval), s_eval, fill).clip(0.0, 1.0)
    return {
        f"{name}_raw": s_eval[eval_pos],
        f"{name}_iso_diso": fit_binary_isotonic(s_cal_src[iso_pos], B_iso).predict(s_eval[eval_pos]),
        f"{name}_iso_dcal": fit_binary_isotonic(s_cal_src[cal_pos], B_cal).predict(s_eval[eval_pos]),
    }


def run_in_distribution(job):
    model, dataset, n_seeds, outputs_dir = job
    try:
        data = BASCData.load(dataset, model, outputs_dir=outputs_dir, verbose=False)
    except FileNotFoundError:
        return []
    anchor_sem = load_anchor_semantic(data, outputs_dir=outputs_dir, verbose=False)
    scores = _token_prob_scores(data)
    max_k = data.k_rollouts
    rows = []
    for seed in range(n_seeds):
        score_pos, iso_pos, test_pos = three_way_split(data.n_queries, seed=seed)
        # Same draws as the main sweep (basc.experiment.run_id_trial).
        score = draw_anchor_set(data, score_pos, max_k - 1, seed=seed * 3 + 1, anchor_semantic=anchor_sem)
        iso = draw_anchor_set(data, iso_pos, 1, seed=seed * 3 + 2, anchor_semantic=anchor_sem)
        test = draw_anchor_set(data, test_pos, max_k - 1, seed=seed * 3 + 3, anchor_semantic=anchor_sem)
        B_iso = iso.agreement[:, 0].astype(np.int64)
        B_cal = np.concatenate([score.agreement[:, 0], iso.agreement[:, 0]]).astype(np.int64)
        cal_pos = np.concatenate([score_pos, iso_pos])
        for name, s in scores.items():
            for method, p in _fit_variants(name, s, s, iso_pos, cal_pos, test_pos, B_iso, B_cal).items():
                m = bundle_all_metrics(p, test.correct, test.agree[:, 0].astype(np.int64))
                rows.append({**{k: v for k, v in m.items() if isinstance(v, (int, float))},
                             "method": method, "model": model, "dataset": dataset,
                             "seed": seed, "accuracy": float(test.correct.mean())})
    return rows


def _shift_rows(train, test, train_sem, test_sem, n_seeds, train_ids=None, test_ids=None):
    tr_scores, te_scores = _token_prob_scores(train), _token_prob_scores(test)
    train_pos = positions_for_ids(train, train_ids)
    test_pos = positions_for_ids(test, test_ids)
    max_k = train.k_rollouts
    out = []
    for seed in range(n_seeds):
        score_pos, iso_pos = split_train_for_shift(train_pos, seed)
        # Same draws as the shift sweep (basc.experiment.run_shift_trial).
        score = draw_anchor_set(train, score_pos, max_k - 1, seed=seed * 3 + 1, anchor_semantic=train_sem)
        iso = draw_anchor_set(train, iso_pos, 1, seed=seed * 3 + 2, anchor_semantic=train_sem)
        te = draw_anchor_set(test, test_pos, 1, seed=seed * 3 + 3, anchor_semantic=test_sem)
        B_iso = iso.agreement[:, 0].astype(np.int64)
        B_cal = np.concatenate([score.agreement[:, 0], iso.agreement[:, 0]]).astype(np.int64)
        cal_pos = np.concatenate([score_pos, iso_pos])
        for name in ("tp", "atp"):
            preds = _fit_variants(name, tr_scores[name], te_scores[name], iso_pos, cal_pos, test_pos,
                                  B_iso, B_cal)
            for method, p in preds.items():
                m = correctness_metrics(p, te.correct)
                out.append({k: m[k] for k in ("ECE1", "ECE2", "Brier", "AUROC")}
                           | {"method": method, "seed": seed, "n_cal": int(len(cal_pos)), "n_test": int(te.n)})
    return out


def run_shift(job):
    shift, model, n_seeds, outputs_dir = job
    cache = {}

    def get(ds):
        if ds not in cache:
            try:
                d = BASCData.load(ds, model, outputs_dir=outputs_dir, verbose=False)
                cache[ds] = (d, load_anchor_semantic(d, outputs_dir=outputs_dir, verbose=False))
            except FileNotFoundError:
                cache[ds] = None
        return cache[ds]

    rows = []
    if shift == "language":
        got = get("polymath")
        if got is None:
            return rows
        d, sem = got
        for tr_l, te_l in LANGUAGE_SPLITS.values():
            for r in _shift_rows(d, d, sem, sem, n_seeds, lang_ids(tr_l), lang_ids(te_l)):
                rows.append({**r, "shift": "language", "model": model,
                             "train": lang_label(tr_l), "test": lang_label(te_l)})
    else:
        for tr_ds, te_ds in (MATH_PAIRS if shift == "math" else QA_PAIRS):
            a, b = get(tr_ds), get(te_ds)
            if a is None or b is None:
                continue
            for r in _shift_rows(a[0], b[0], a[1], b[1], n_seeds):
                rows.append({**r, "shift": shift, "model": model, "train": tr_ds, "test": te_ds})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=("in_distribution", "shift"), required=True)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--outputs_dir", default=None)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    t0 = time.time()
    if args.setting == "in_distribution":
        fn, jobs = run_in_distribution, [(m, d, args.n_seeds, args.outputs_dir) for m, d in
                                         [(m, d) for m in roster.MODELS for d in roster.datasets_for(m)]]
    else:
        fn, jobs = run_shift, [(s, m, args.n_seeds, args.outputs_dir)
                               for s in ("language", "math", "qa") for m in roster.MODELS]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(fn, jobs) for r in rs]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[write] {args.out_csv}  rows={len(rows)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
