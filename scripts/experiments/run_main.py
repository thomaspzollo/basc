#!/usr/bin/env python3
"""Main in-distribution experiment (Table 1, Figures 2 and 3(a), appendix per-model /
per-dataset results and reliability diagrams).

For every (model, dataset) and seed, the 1000 questions are split 20/20/60 into
D_score / D_iso / D_test, and every method is scored on the deployed T=0.6
answers of D_test:
  - calibration-set methods (basc.experiment): BASC-L, LR-SC, LR-SSC, the
    supervised reference and TT-SC;
  - single-generation baselines: TP, ATP and VC.
Rows hold ECE1/ECE2/MCE/NLL/Brier/AUROC and pooled-reliability bins; BASC-style
rows also hold calibration against a held-out probe (agree_*).

Also used for the black-box variant (--variant gemma, BASC-L-bb) and the
probe-temperature ablation (--pool_temp/--pool_k with --test_pool_temp 0.7
--test_pool_k 100); see scripts/06_experiments.sh.

Example:
    python scripts/experiments/run_main.py --out_csv results/main.csv
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

import pandas as pd

from basc import paths, roster
from basc.baselines import load_ssc_c_matrix
from basc.data import BASCData
from basc.experiment import external_baselines, run_id_trial
from basc.metrics import bundle_all_metrics, correctness_metrics, reliability_bins
from basc.semantic import load_anchor_semantic


def _scalars(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}


def run_dataset(model: str, dataset: str, args) -> pd.DataFrame | None:
    t0 = time.time()
    tp_temp = args.test_pool_temp if args.test_pool_temp is not None else args.pool_temp
    tp_k = args.test_pool_k if args.test_pool_k is not None else args.pool_k
    try:
        data = BASCData.load(dataset, model, n=args.n, pool_temp=args.pool_temp, pool_k=args.pool_k,
                             test_pool_temp=args.test_pool_temp, test_pool_k=args.test_pool_k,
                             variant=args.variant, outputs_dir=args.outputs_dir)
    except FileNotFoundError as e:
        print(f"[skip] {model}/{dataset}: {e}")
        return None
    anchor_sem = load_anchor_semantic(data, n=args.n, pool_temp=args.pool_temp, pool_k=args.pool_k,
                                      outputs_dir=args.outputs_dir)
    ssc_c_mat = ssc_answers = None
    if args.ssc_k_variants:
        ssc_c_mat, ssc_answers = load_ssc_c_matrix(model, dataset, data.ids, n=args.n,
                                                   pool_temp=args.pool_temp, pool_k=args.pool_k,
                                                   outputs_dir=args.outputs_dir)
    vc_path = paths.verbal_confidence(model, dataset, stem_idx=6, n=args.n, pool_temp=tp_temp,
                                      pool_k=tp_k, outputs_dir=args.outputs_dir)
    common = {"dataset": dataset, "model": model, "n_queries": data.n_queries}

    frames = []
    for seed in range(args.n_seeds):
        scores, correct, agree, test_ids, sem_agree = run_id_trial(
            data, seed,
            k_variants=args.k_variants, ridge_k_variants=args.ridge_k_variants,
            ssc_k_variants=args.ssc_k_variants, ssc_c_mat=ssc_c_mat, ssc_answers=ssc_answers,
            anchor_semantic=anchor_sem,
        )
        rows = []
        for method, preds in scores.items():
            probe = sem_agree if ("_sem_k" in method and sem_agree is not None) else agree
            rows.append({"method": method, **_scalars(bundle_all_metrics(preds, correct, probe)),
                         **reliability_bins(preds, correct)})
        ext, ext_correct = external_baselines(data, test_ids, vc_path=vc_path)
        for method, preds in ext.items():
            rows.append({"method": method, **correctness_metrics(preds, ext_correct),
                         **reliability_bins(preds, ext_correct)})
        df = pd.DataFrame(rows)
        df["seed"] = seed
        df["accuracy"] = float(correct.mean())
        df["n_test"] = int(len(correct))
        frames.append(df.assign(**common))
    print(f"[{model} / {dataset}] {time.time() - t0:.1f}s")
    return pd.concat(frames, ignore_index=True)


def run_model(model: str, args) -> pd.DataFrame | None:
    datasets = args.datasets or roster.datasets_for(model)
    frames = [f for f in (run_dataset(model, d, args) for d in datasets) if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=roster.MODELS)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="default: the paper's datasets for each model (basc.roster.datasets_for)")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=paths.N)
    ap.add_argument("--pool_temp", type=float, default=paths.POOL_TEMP, help="probe pool temperature")
    ap.add_argument("--pool_k", type=int, default=paths.POOL_K, help="probe pool size per question")
    ap.add_argument("--test_pool_temp", type=float, default=None,
                    help="pool stem of the deployed-response files (default: --pool_temp); set to 0.7 "
                         "with --test_pool_k 100 to reuse the canonical deployed responses")
    ap.add_argument("--test_pool_k", type=int, default=None)
    ap.add_argument("--k_variants", nargs="+", type=int, default=[2, 5, 10, 20, 100],
                    help="BASC-L decodes per Stage-1 question")
    ap.add_argument("--ridge_k_variants", nargs="+", type=int, default=[2, 5, 10, 20, 100],
                    help="LR-SC decodes per question")
    ap.add_argument("--ssc_k_variants", nargs="*", type=int, default=[2, 5, 10, 20, 100],
                    help="LR-SSC decodes per question (skipped beyond the SSC file's coverage)")
    ap.add_argument("--variant", choices=("own", "gemma"), default="own",
                    help="embedding backbone: the reasoning model ('own') or Gemma-3-4B-it ('gemma')")
    ap.add_argument("--outputs_dir", default=None, help="default: basc.paths.OUTPUTS_DIR")
    ap.add_argument("--workers", type=int, default=None, help="parallel models (default: one per model)")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    t0 = time.time()
    with ProcessPoolExecutor(args.workers or len(args.models)) as ex:
        results = dict(zip(args.models, ex.map(run_model, args.models, [args] * len(args.models))))
    # Concatenate in the order of the per-model file names of the original sweep.
    order = sorted(args.models, key=lambda m: m.replace("/", "_").replace(".", "_"))
    frames = [results[m] for m in order if results[m] is not None]
    if not frames:
        raise SystemExit("no results produced")
    full = pd.concat(frames, ignore_index=True)
    full["n"] = args.n
    full["pool_temp"] = args.pool_temp
    full["pool_k"] = args.pool_k
    full["variant"] = args.variant
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    full.to_csv(args.out_csv, index=False)
    print(f"[write] {args.out_csv}  rows={len(full)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
