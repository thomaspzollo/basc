#!/usr/bin/env python3
"""Slim raw generation pickles into the per-question tables used downstream.

Probe pool (n_{n}_temp_{T}_k_{k}.pkl) ->
  *_SLIM_train_df.pkl       id, answer                         (BASCData)
  *_ROLLOUTS_train_df.pkl   id, sample_idx, answer, response   (embedding, SSC)
Deployed responses (n_{n}_temp_0.6_k_1.pkl), unless --pool_only ->
  *_SLIM_test_df_temp_0.6_k_1.pkl
  *_out_df_temp_0.6_k_1.csv  one row per question: question, ground_truth,
      answer, response, avg_logprobs (mean token log-prob of the response),
      ans_logprobs (mean log-prob of the \\boxed{} answer tokens),
      num_output_tokens, correct, consistency (majority-answer frequency in
      the pool)

Example:
    python scripts/pipeline/process_generations.py --model Qwen/Qwen3-8B --dataset gsm8k
    python scripts/pipeline/process_generations.py --model Qwen/Qwen3-1.7B --dataset gsm8k \\
        --pool_temp 0.5 --pool_k 20 --pool_only
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle as pkl
import time
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from basc import paths
from basc.answers import compute_correctness


def _load_pickle(path) -> list:
    print(f"[load] {path}  ({os.path.getsize(path) / 1e9:.1f} GB)")
    t0 = time.time()
    with open(path, "rb", buffering=1024 * 1024 * 64) as f:
        rows = pkl.load(f)
    print(f"[load] {len(rows):,} rows in {time.time() - t0:.0f}s")
    return rows


def _iter_steps(lp_steps) -> Iterable[Tuple[str, float]]:
    """(decoded token, log-prob) of the sampled token at each generation step."""
    for step in lp_steps:
        if not isinstance(step, dict) or len(step) == 0:
            continue
        v = next(iter(step.values()))
        if hasattr(v, "logprob") and hasattr(v, "decoded_token"):
            yield str(v.decoded_token).strip(), float(v.logprob)
        elif isinstance(v, dict) and "decoded_token" in v and "logprob" in v:
            yield str(v["decoded_token"]).strip(), float(v["logprob"])
        else:
            try:
                yield str(getattr(v, "decoded_token", "")).strip(), float(getattr(v, "logprob", 0.0))
            except Exception:
                continue


def extract_lp_features(lp_steps) -> Tuple[float, float, int]:
    """(mean log-prob over all tokens, mean log-prob of the answer span, #tokens).

    The answer span starts two tokens after a token equal to "boxed" (skipping
    "{") and ends before the first token containing "}". Responses without
    "boxed" get -100; an empty span gets 0.
    """
    total, n_steps = 0.0, 0
    started, skip, span_sum, span_n = False, 0, 0.0, 0
    for tok, lp in _iter_steps(lp_steps):
        total += lp
        n_steps += 1
        if not started:
            if tok == "boxed":
                started, skip = True, 2
            continue
        if skip > 0:
            skip -= 1
            continue
        if "}" in tok:
            break
        span_sum += lp
        span_n += 1
    avg_lp = total / max(n_steps, 1)
    ans_lp = -100.0 if not started else (0.0 if span_n == 0 else span_sum / span_n)
    return avg_lp, ans_lp, n_steps


def process_pool(args) -> pd.DataFrame:
    kw = dict(n=args.n, temp=args.pool_temp, k=args.pool_k, outputs_dir=args.outputs_dir)
    answers_path = paths.pool_answers(args.model, args.dataset, **kw)
    responses_path = paths.pool_responses(args.model, args.dataset, **kw)
    if answers_path.exists() and responses_path.exists() and not args.force:
        print(f"[pool] exists: {answers_path}")
        return pd.read_pickle(answers_path)

    rows = _load_pickle(paths.raw_generations(args.model, args.dataset, **kw))
    df = pd.DataFrame(rows)[["id", "sample_idx", "answer", "response"]]
    del rows
    gc.collect()
    df["id"] = df["id"].astype(int)
    df["sample_idx"] = df["sample_idx"].astype(int)
    df["answer"] = df["answer"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)

    # Generation order is (id, sample_idx) already; BASCData relies on it.
    answers = df[["id", "answer"]].copy()
    answers.to_pickle(answers_path)
    print(f"[write] {answers_path}")

    responses = df.sort_values(["id", "sample_idx"]).reset_index(drop=True)
    per_id = responses.groupby("id")["sample_idx"].size()
    if per_id.nunique() != 1:
        print(f"[warn] uneven pool size per id: min={per_id.min()} max={per_id.max()}")
    responses.to_pickle(responses_path)
    print(f"[write] {responses_path}")
    return answers


def process_test(args, pool_answers: pd.DataFrame) -> None:
    kw = dict(n=args.n, pool_temp=args.pool_temp, pool_k=args.pool_k, test_temp=args.test_temp,
              test_k=1, outputs_dir=args.outputs_dir)
    rows = _load_pickle(paths.raw_generations(args.model, args.dataset, n=args.n, temp=args.test_temp,
                                              k=1, outputs_dir=args.outputs_dir))
    test_df = pd.DataFrame(rows)[["id", "question", "ground_truth", "answer", "response", "logprobs"]].copy()
    del rows
    test_df["id"] = test_df["id"].astype(int)
    test_df["answer"] = test_df["answer"].fillna("").astype(str)
    test_df = test_df.sort_values(["id"]).drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    feats = [extract_lp_features(lp) for lp in tqdm(test_df["logprobs"].tolist(), desc="[test] logprob features")]
    test_df["avg_logprobs"] = np.array([f[0] for f in feats], dtype=np.float64)
    test_df["ans_logprobs"] = np.array([f[1] for f in feats], dtype=np.float64)
    test_df["num_output_tokens"] = np.array([f[2] for f in feats], dtype=np.int32)
    test_df = test_df.drop(columns=["logprobs"])
    test_df.to_pickle(paths.test_slim(args.model, args.dataset, **kw))

    # Empirical self-consistency: frequency of the pool's majority answer.
    counts = pool_answers.groupby("id")["answer"].value_counts(dropna=False)
    consistency = (counts.groupby(level=0).max() / pool_answers.groupby("id")["answer"].size()).sort_index()

    out_df = test_df.copy()
    out_df["correct"] = compute_correctness(test_df, args.dataset)
    out_df["consistency"] = test_df["id"].map(consistency).fillna(0.0).to_numpy(dtype=np.float64)
    out_path = paths.test_responses(args.model, args.dataset, **kw)
    out_df.to_csv(out_path, index=False)
    print(f"[write] {out_path}  accuracy={out_df['correct'].mean():.3f}  "
          f"empty answers={(out_df['answer'] == '').sum()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=paths.N)
    ap.add_argument("--pool_temp", type=float, default=paths.POOL_TEMP)
    ap.add_argument("--pool_k", type=int, default=paths.POOL_K)
    ap.add_argument("--test_temp", type=float, default=paths.TEST_TEMP)
    ap.add_argument("--pool_only", action="store_true", help="skip the deployed-response table")
    ap.add_argument("--force", action="store_true", help="rebuild pool slims even if they exist")
    ap.add_argument("--outputs_dir", default=None)
    args = ap.parse_args()

    pool_answers = process_pool(args)
    if not args.pool_only:
        process_test(args, pool_answers)


if __name__ == "__main__":
    main()
