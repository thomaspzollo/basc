#!/usr/bin/env python3
"""Per-response P(Yes) scores for the RR-SSC baseline (Huang et al., 2025).

For the first k_ssc=32 probe-pool responses of every question, append
"Is the answer correct? (Yes/No)" to the (question, response) conversation and
read the probability mass on first tokens containing "Yes" (top-20 log-probs)
as c_i. The soft self-consistency target of each response is

    SSC(y_j) = sum_{i: a_i == a_j} c_i / sum_i c_i

with answers grouped by exact match of the cleaned answer string.

Output: *_ROLLOUTS_ssc_k32.csv
        {id, sample_idx, answer, c, ssc, cluster_id, cluster_size, n_scored}
RR-SSC recomputes SSC over the first k responses from `c` (basc.baselines).
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from basc import paths
from basc.answers import clean_answer
from basc.generation import GenerateConfig, load_vllm_generate

CONFIRM_QUESTION = "Is the answer correct? (Yes/No)"
YES_SUBSTR = "Yes"


def _mass_by_decoded_substr(lp: dict, substr: str) -> float:
    """Total probability of top-k first tokens whose decoded string contains `substr`."""
    return sum(float(np.exp(e.logprob)) for e in lp.values()
               if getattr(e, "decoded_token", None) is not None and substr in e.decoded_token)


def build_confirm_prompt(tokenizer, question: str, response: str) -> str:
    messages = [
        {"role": "user", "content": str(question)},
        {"role": "assistant", "content": str(response)},
        {"role": "user", "content": CONFIRM_QUESTION},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Close an open <think> block so the next token is the direct Yes/No answer.
    if prompt.rfind("<think>") != -1 and prompt.rfind("</think>") < prompt.rfind("<think>"):
        prompt += "</think>\n\n"
    return prompt


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    kw = dict(n=args.n, temp=paths.POOL_TEMP, k=paths.POOL_K, outputs_dir=args.outputs_dir)
    out_csv = paths.ssc_scores(args.model, args.dataset, k_ssc=args.k_ssc, **kw)
    df = pd.read_pickle(paths.pool_responses(args.model, args.dataset, **kw))
    df = df[df["sample_idx"] < args.k_ssc].sort_values(["id", "sample_idx"]).reset_index(drop=True)
    test_csv = paths.test_responses(args.model, args.dataset, n=args.n, pool_temp=paths.POOL_TEMP,
                                    pool_k=paths.POOL_K, outputs_dir=args.outputs_dir)
    q_map = pd.read_csv(test_csv, usecols=["id", "question"]).assign(
        id=lambda d: d["id"].astype(int)).set_index("id")["question"].to_dict()

    cfg = GenerateConfig(model=args.model, n_device=args.n_device, temperature=1.0, top_p=1.0,
                         max_new_tokens=1, seed=args.seed, logprobs=20, min_p=0.0, top_k=-1)
    tokenizer, llm, sampling = load_vllm_generate(cfg)
    prompts = [build_confirm_prompt(tokenizer, str(q_map[int(r["id"])]), str(r["response"]))
               for _, r in df.iterrows()]

    c_vals = np.full(len(prompts), np.nan)
    for start in range(0, len(prompts), args.batch_size):
        for i, res in enumerate(llm.generate(prompts[start:start + args.batch_size], sampling)):
            if res.outputs and res.outputs[0].logprobs:
                c_vals[start + i] = _mass_by_decoded_substr(res.outputs[0].logprobs[0], YES_SUBSTR)
    c_vals[~np.isfinite(c_vals)] = 0.0
    df["c"] = c_vals

    ssc = np.zeros(len(df))
    cluster_id = np.zeros(len(df), dtype=np.int64)
    cluster_size = np.zeros(len(df), dtype=np.int64)
    n_scored = np.zeros(len(df), dtype=np.int64)
    for _, idx in df.groupby("id", sort=False).indices.items():
        idx = np.asarray(idx, dtype=int)
        keys: dict[str, int] = {}
        cids = np.array([keys.setdefault(clean_answer(a, lower=True), len(keys))
                         for a in df["answer"].iloc[idx].astype(str)], dtype=np.int64)
        c_q = c_vals[idx]
        denom = float(c_q.sum())
        if denom > 0:
            mass = defaultdict(float)
            for cid, cv in zip(cids, c_q):
                mass[int(cid)] += float(cv)
            ssc[idx] = [mass[int(cid)] / denom for cid in cids]
        sizes = dict(zip(*np.unique(cids, return_counts=True)))
        cluster_id[idx], cluster_size[idx], n_scored[idx] = cids, [sizes[c] for c in cids], len(idx)
    df["ssc"], df["cluster_id"], df["cluster_size"], df["n_scored"] = ssc, cluster_id, cluster_size, n_scored

    df[["id", "sample_idx", "answer", "c", "ssc", "cluster_id", "cluster_size", "n_scored"]].to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k_ssc", type=int, default=32, help="scored responses per question (Huang et al. use 32)")
    ap.add_argument("--n", type=int, default=paths.N)
    ap.add_argument("--n_device", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outputs_dir", default=None)
    main(ap.parse_args())
