#!/usr/bin/env python3
"""Verbalized-confidence (VC) baseline.

The reasoning model is shown each question and its deployed answer (not the
reasoning trace) with one of ten prompt stems (basc.verbal_confidence) and asked
for the probability that the answer is correct, as \\boxed{p}. Stem 6 is used in
the main experiments; all ten are needed for the prompt-selection appendix
table.

Output: *_out_df_temp_0.6_k_1_verbal_conf_stem_{i}.csv
        {id, verbal_confidence (NaN if unparseable), verbal_confidence_raw}
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from basc import paths
from basc.answers import clean_answer
from basc.generation import GenerateConfig, load_vllm_generate
from basc.verbal_confidence import VERBAL_CONF_STEM_IDX, build_verbal_conf_prompt, parse_prob


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    in_csv = paths.test_responses(args.model, args.dataset, n=args.n, pool_temp=paths.POOL_TEMP,
                                  pool_k=paths.POOL_K, outputs_dir=args.outputs_dir)
    out_csv = paths.verbal_confidence(args.model, args.dataset, stem_idx=args.stem_idx, n=args.n,
                                      outputs_dir=args.outputs_dir)
    df = pd.read_csv(in_csv)

    cfg = GenerateConfig(model=args.model, n_device=args.n_device, temperature=0.6, top_p=0.95,
                         max_new_tokens=2048, seed=args.seed, logprobs=0, min_p=0.0, top_k=20)
    tokenizer, llm, sampling = load_vllm_generate(cfg)
    prompts = [
        build_verbal_conf_prompt(tokenizer, args.model, question=str(row["question"]),
                                 answer=clean_answer(str(row["answer"]), lower=False), stem_idx=args.stem_idx)
        for _, row in df.iterrows()
    ]
    print("[vc] example prompt:\n" + prompts[0])

    raw = [""] * len(prompts)
    for start in range(0, len(prompts), args.batch_size):
        results = llm.generate(prompts[start:start + args.batch_size], sampling)
        for i, res in enumerate(results):
            raw[start + i] = res.outputs[0].text if res.outputs else ""
    probs = [parse_prob(t) for t in raw]
    print(f"[vc] could not parse {sum(p is None for p in probs)}/{len(probs)} outputs")

    os.makedirs(out_csv.parent, exist_ok=True)
    pd.DataFrame({"id": df["id"].astype(str).tolist(), "verbal_confidence": probs,
                  "verbal_confidence_raw": raw}).to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stem_idx", type=int, default=VERBAL_CONF_STEM_IDX)
    ap.add_argument("--n", type=int, default=paths.N)
    ap.add_argument("--n_device", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outputs_dir", default=None)
    main(ap.parse_args())
