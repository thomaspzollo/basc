#!/usr/bin/env python3
"""Sample k responses per question with vLLM and pickle them with token log-probs.

The paper uses, for every (model, dataset):
  --k 100 --temperature 0.7   probe pool
  --k 1   --temperature 0.6   deployed response (the one being calibrated)
and, for the temperature ablation, --k 20 at temperatures 0.5 and 0.9.
All runs use top-p 0.95, top-k 20, up to 32768 new tokens and seed 12345.

Output: {outputs}/{model_short}/{dataset}/n_{n}_temp_{T}_k_{k}.pkl, a list of
dicts (id, question, ground_truth, prompt, sample_idx, response, answer,
logprobs). Pool pickles can exceed 10 GB; process_generations.py slims them.
"""
from __future__ import annotations

import argparse
import os
import pickle as pkl
import random

import numpy as np
import torch

from basc import paths
from basc.datasets import load_dataset_items
from basc.generation import GenerateConfig, gather_prompts, generate_k, load_vllm_generate, rows_from_generation


def main(args):
    out_path = paths.raw_generations(args.model, args.dataset, n=args.n, temp=args.temperature, k=args.k,
                                     outputs_dir=args.outputs_dir)
    os.makedirs(out_path.parent, exist_ok=True)
    print(f"[generate] {args.model} / {args.dataset}  n={args.n} k={args.k} T={args.temperature} -> {out_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = load_dataset_items(args.dataset, n=args.n)
    cfg = GenerateConfig(
        model=args.model, n_device=args.n_device, temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens, seed=args.seed, logprobs=1, min_p=0.0, top_k=20,
    )
    tokenizer, llm, sampling = load_vllm_generate(cfg)
    prompts = gather_prompts(tokenizer, items, args.model)
    print("[generate] example prompt:\n" + prompts[0])

    results = generate_k(llm, prompts, sampling, k=args.k)
    rows, bad_frac = rows_from_generation(items, prompts, results)
    print(f"[generate] {len(rows)} responses; {bad_frac:.1%} without a \\boxed{{}} answer")

    with open(out_path, "wb") as f:
        pkl.dump(rows, f)
    print(f"[write] {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "polymath", "sciq", "trivia_qa", "webq"])
    ap.add_argument("--n", type=int, default=paths.N, help="number of questions")
    ap.add_argument("--k", type=int, required=True, help="responses per question")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n_device", type=int, default=1, help="tensor-parallel GPUs")
    ap.add_argument("--outputs_dir", default=None)
    main(ap.parse_args())
