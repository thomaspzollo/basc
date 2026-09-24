#!/usr/bin/env python3
"""Distribution shift (Table 2 and appendix shift figures).

Every calibration-set method is fit on a training domain and evaluated on the
deployed answers of a test domain:
  language  PolyMath, 8 splits of 4 training / 4 test languages
  math      GSM8K <-> PolyMath
  qa        all ordered pairs of SciQ, TriviaQA and WebQ
Single-generation baselines (TP, ATP, VC) need no calibration data and are
scored once per pair on the test domain.

Writes {out_dir}/shift_{language,math,qa}.csv.
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
from basc.experiment import external_baselines, run_shift_trial
from basc.metrics import correctness_metrics
from basc.semantic import load_anchor_semantic
from basc.shift import LANGUAGE_SPLITS, MATH_PAIRS, QA_PAIRS, lang_ids, lang_label

SEEDS = range(10)
TRIAL_KW = dict(ridge_k_variants=(5, 10, 20, 100), ssc_k_variants=(5, 10, 20, 100))


class _Cache:
    """Per-model cache of BASCData, semantic and SSC matrices by dataset."""

    def __init__(self, model, outputs_dir):
        self.model, self.outputs_dir, self._d = model, outputs_dir, {}

    def get(self, dataset):
        if dataset not in self._d:
            data = BASCData.load(dataset, self.model, outputs_dir=self.outputs_dir, verbose=False)
            self._d[dataset] = dict(
                data=data,
                sem=load_anchor_semantic(data, outputs_dir=self.outputs_dir, verbose=False),
                ssc=load_ssc_c_matrix(self.model, dataset, data.ids, outputs_dir=self.outputs_dir),
                vc_path=paths.verbal_confidence(self.model, dataset, stem_idx=6, outputs_dir=self.outputs_dir),
            )
        return self._d[dataset]


def _pair_rows(train, test, labels, train_ids=None, test_ids=None):
    rows = []
    for seed in SEEDS:
        scores, correct, _, tids = run_shift_trial(
            train["data"], test["data"], seed=seed, train_id_subset=train_ids, test_id_subset=test_ids,
            ssc_c_mat_train=train["ssc"][0], ssc_answers_train=train["ssc"][1],
            anchor_semantic_train=train["sem"], anchor_semantic_test=test["sem"], **TRIAL_KW,
        )
        for method, s in scores.items():
            rows.append({**correctness_metrics(s, correct), "method": method, **labels, "seed": seed})
    # The test questions do not depend on the seed; baselines are scored once.
    ext, ext_correct = external_baselines(test["data"], tids, vc_path=test["vc_path"], float64_logprobs=True)
    for method, s in ext.items():
        rows.append({**correctness_metrics(s, ext_correct), "method": method, **labels})
    return rows


def run_job(job):
    shift, model, outputs_dir = job
    cache, rows = _Cache(model, outputs_dir), []
    try:
        if shift == "language":
            poly = cache.get("polymath")
            for split_id, (tr, te) in LANGUAGE_SPLITS.items():
                labels = {"model": model, "split": split_id,
                          "train_langs": lang_label(tr), "test_langs": lang_label(te)}
                rows += _pair_rows(poly, poly, labels, lang_ids(tr), lang_ids(te))
        else:
            for train_ds, test_ds in (MATH_PAIRS if shift == "math" else QA_PAIRS):
                if test_ds not in roster.datasets_for(model):
                    continue
                labels = {"model": model, "train_ds": train_ds, "test_ds": test_ds}
                rows += _pair_rows(cache.get(train_ds), cache.get(test_ds), labels)
    except FileNotFoundError as e:
        print(f"[skip] {model} / {shift}: {e}")
    print(f"  done {shift} / {model}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shifts", nargs="+", default=["language", "math", "qa"])
    ap.add_argument("--outputs_dir", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out_dir", default=str(paths.RESULTS_DIR))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for shift in args.shifts:
        t0 = time.time()
        jobs = [(shift, m, args.outputs_dir) for m in roster.MODELS]
        with ProcessPoolExecutor(args.workers) as ex:
            rows = [r for rs in ex.map(run_job, jobs) for r in rs]
        out = os.path.join(args.out_dir, f"shift_{shift}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[write] {out}  rows={len(rows)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
