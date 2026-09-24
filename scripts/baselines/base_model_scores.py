#!/usr/bin/env python3
"""Base-model confidence (appendix: "Base Models for Weak Supervision").

Scores each deployed response of a Qwen3 reasoning model under its paired
pre-trained (Base) model, following Luo et al. (2025) and Tan et al. (2026):
  base_response_*  mean token log-prob of the full response (with reasoning)
                   after the reasoning model's chat prompt
  base_answer_*    mean token log-prob of the extracted answer after the plain
                   prompt "Question: {q}\\nAnswer: "
*_avg_prob = exp(*_avg_logprob) is the confidence score.

Output: {model_short_lower}_{dataset}_with_base_targets.csv (the deployed-
response table plus the base_* columns).
"""
from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from basc import paths, roster
from basc.generation import QA_PROMPT_STEM


def _s(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def response_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": f"{QA_PROMPT_STEM}\nQuestion: {question}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def answer_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer: "


@torch.no_grad()
def mean_continuation_logprob(model, tokenizer, prompts: List[str], continuations: List[str],
                              batch_size: int) -> Tuple[List[float], List[int]]:
    """Mean log-prob of each continuation's tokens given its prompt."""
    device = next(model.parameters()).device
    mean_lps, ns = [], []
    for start in tqdm(range(0, len(prompts), batch_size), desc="scoring"):
        p_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts[start:start + batch_size]]
        c_ids = [tokenizer.encode(c, add_special_tokens=False) for c in continuations[start:start + batch_size]]
        seqs = [p + c for p, c in zip(p_ids, c_ids)]
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[i, :len(seq)] = 1
        log_probs = F.log_softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1)
        for i, (p, c) in enumerate(zip(p_ids, c_ids)):
            if not c:
                mean_lps.append(float("nan"))
                ns.append(0)
                continue
            total = sum(log_probs[i, len(p) + j - 1, tok].item() for j, tok in enumerate(c))
            mean_lps.append(total / len(c))
            ns.append(len(c))
    return mean_lps, ns


def main(args):
    base_model = roster.BASE_MODELS[args.model]
    in_csv = paths.test_responses(args.model, args.dataset, n=paths.N, pool_temp=paths.POOL_TEMP,
                                  pool_k=paths.POOL_K, outputs_dir=args.outputs_dir)
    out_csv = paths.base_model_scores(args.model, args.dataset, outputs_dir=args.outputs_dir)
    df = pd.read_csv(in_csv)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=dtype,
                                                 device_map="auto").eval()

    questions = [_s(q) for q in df["question"].tolist()]
    resp_lps, resp_ns = mean_continuation_logprob(
        model, tokenizer, [response_prompt(tokenizer, q) for q in questions],
        [_s(r) for r in df["response"].tolist()], args.batch_size)
    ans_lps, ans_ns = mean_continuation_logprob(
        model, tokenizer, [answer_prompt(q) for q in questions],
        [str(a) for a in df["answer"].tolist()], args.batch_size)

    df["base_response_avg_logprob"] = resp_lps
    df["base_response_avg_prob"] = [math.exp(x) if not math.isnan(x) else float("nan") for x in resp_lps]
    df["base_response_num_tokens"] = resp_ns
    df["base_answer_avg_logprob"] = ans_lps
    df["base_answer_avg_prob"] = [math.exp(x) if not math.isnan(x) else float("nan") for x in ans_lps]
    df["base_answer_num_tokens"] = ans_ns
    df.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(roster.BASE_MODELS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--outputs_dir", default=None)
    main(ap.parse_args())
