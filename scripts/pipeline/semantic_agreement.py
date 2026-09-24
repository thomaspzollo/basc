#!/usr/bin/env python3
"""Precompute semantic (bidirectional-entailment) agreement for a QA dataset.

For each question, compares the deployed T=0.6 answer with every probe-pool
response using an NLI cross-encoder (cross-encoder/nli-deberta-v3-base) and
writes an (N, K) boolean matrix:
    *_ANCHOR_T0.6_semantic_agree.npz   {ids, semantic_agree, nli_model_name}

Example:
    python scripts/pipeline/semantic_agreement.py --model Qwen/Qwen3-8B --dataset sciq --device cuda:0 --fp16
    # temperature-ablation pool (reuses the canonical deployed answers):
    python scripts/pipeline/semantic_agreement.py --model Qwen/Qwen3-1.7B --dataset sciq \\
        --pool_temp 0.5 --pool_k 20 --test_pool_temp 0.7 --test_pool_k 100
"""
from __future__ import annotations

import argparse
import time

from basc import paths
from basc.data import BASCData
from basc.semantic import NLI_MODEL_NAME_DEFAULT, build_anchor_mutual_entailment, save_semantic_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["sciq", "trivia_qa", "webq"])
    ap.add_argument("--n", type=int, default=paths.N)
    ap.add_argument("--pool_temp", type=float, default=paths.POOL_TEMP)
    ap.add_argument("--pool_k", type=int, default=paths.POOL_K)
    ap.add_argument("--test_pool_temp", type=float, default=None)
    ap.add_argument("--test_pool_k", type=int, default=None)
    ap.add_argument("--nli_model", default=NLI_MODEL_NAME_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--outputs_dir", default=None)
    args = ap.parse_args()

    out_path = paths.anchor_semantic_agreement(args.model, args.dataset, n=args.n, temp=args.pool_temp,
                                               k=args.pool_k, outputs_dir=args.outputs_dir)
    if out_path.exists() and not args.force:
        print(f"[skip] exists: {out_path}")
        return

    data = BASCData.load(args.dataset, args.model, n=args.n, pool_temp=args.pool_temp, pool_k=args.pool_k,
                         test_pool_temp=args.test_pool_temp, test_pool_k=args.test_pool_k,
                         outputs_dir=args.outputs_dir)
    q_by_id = dict(zip(data.out_df["id"].astype(int), data.out_df["question"]))
    questions = [q_by_id[int(i)] for i in data.ids]

    t0 = time.time()
    mat = build_anchor_mutual_entailment(questions, data.test_answers, data.answers,
                                         nli_model_name=args.nli_model, batch_size=args.batch_size,
                                         device=args.device, fp16=args.fp16)
    save_semantic_cache(out_path, ids=data.ids, semantic_agree=mat, nli_model_name=args.nli_model)
    print(f"[write] {out_path}  shape={mat.shape}  agreement rate={mat.mean():.3f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
