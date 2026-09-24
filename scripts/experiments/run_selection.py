#!/usr/bin/env python3
"""Selective prediction (Figure 4 and appendix selection figures).

Same splits and methods as run_main.py. For each method, test questions are
ranked by confidence and the least-confident fraction a in {0, 0.05, ..., 0.9}
is abstained on; we record accuracy and mean confidence of the selected and
rejected sets.

Example:
    python scripts/experiments/run_selection.py --out_csv results/selection.csv
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

from basc import paths, roster
from basc.baselines import load_ssc_c_matrix
from basc.data import BASCData
from basc.experiment import external_baselines, run_id_trial
from basc.metrics import selective_acc_at_abstention
from basc.semantic import load_anchor_semantic

ABSTENTION_RATES = np.arange(0, 95, 5) / 100.0


def _curve(scores, correct, **cols) -> pd.DataFrame:
    df = selective_acc_at_abstention(scores, correct, ABSTENTION_RATES)
    for k, v in cols.items():
        df[k] = v
    df["full_acc"] = float(np.mean(correct))
    df["sel_imp"] = (df["sel_acc"] - df["full_acc"]) / df["full_acc"]
    return df


def run_combo(job) -> list[pd.DataFrame]:
    model, dataset, n_seeds, outputs_dir = job
    try:
        data = BASCData.load(dataset, model, outputs_dir=outputs_dir, verbose=False)
    except FileNotFoundError as e:
        print(f"[skip] {model}/{dataset}: {e}")
        return []
    anchor_sem = load_anchor_semantic(data, outputs_dir=outputs_dir)
    ssc_c_mat, ssc_answers = load_ssc_c_matrix(model, dataset, data.ids, outputs_dir=outputs_dir)
    vc_path = paths.verbal_confidence(model, dataset, stem_idx=6, outputs_dir=outputs_dir)
    frames = []
    for seed in range(n_seeds):
        scores, correct, _, test_ids, _ = run_id_trial(
            data, seed, ridge_k_variants=(5, 10, 20, 100), ssc_k_variants=(5, 10, 20, 100),
            ssc_c_mat=ssc_c_mat, ssc_answers=ssc_answers, anchor_semantic=anchor_sem,
        )
        for method, s in scores.items():
            frames.append(_curve(s, correct, method=method, dataset=dataset, model=model, seed=seed))
        ext, ext_correct = external_baselines(data, test_ids, vc_path=vc_path, float64_logprobs=True)
        for method, s in ext.items():
            frames.append(_curve(s, ext_correct, method=method, dataset=dataset, model=model, seed=seed))
    print(f"  done {model} / {dataset}")
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--outputs_dir", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    t0 = time.time()
    jobs = [(m, d, args.n_seeds, args.outputs_dir) for m, d in roster.combos()]
    with ProcessPoolExecutor(args.workers) as ex:
        frames = [f for fs in ex.map(run_combo, jobs) for f in fs]
    full = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    full.to_csv(args.out_csv, index=False)
    print(f"[write] {args.out_csv}  rows={len(full)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
