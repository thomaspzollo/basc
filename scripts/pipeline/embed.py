#!/usr/bin/env python3
"""Embed (question, response) pairs with a vLLM embedding model.

  --target test     the deployed response of every question
                    -> *_out_df_temp_0.6_k_1_question_response[_gemma]_embeddings.npz
                       {ids (str), embeddings}
  --target pool     every probe-pool response
                    -> *_ROLLOUTS_qr[_gemma]_embeddings.npz
                       {ids, sample_idx, embeddings}, ordered by (id, sample_idx)

The embedding model is the reasoning model itself (--variant own; BASC-L) or
Gemma-3-4B-it (--variant gemma; black-box BASC-L-bb). Each text is
"QUESTION:\\n{question}\\n\\nRESPONSE:\\n{response}" (question only if the
response is empty).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from basc import paths
from basc.generation import embed_texts, load_vllm_embed


def build_text(question, response) -> str:
    q = "" if question is None else str(question)
    r = "" if response is None else str(response)
    return f"QUESTION:\n{q}\n\nRESPONSE:\n{r}" if r.strip() else f"QUESTION:\n{q}"


def main(args):
    embed_model = args.model if args.variant == "own" else paths.BLACKBOX_EMBED_MODEL
    test_csv = paths.test_responses(args.model, args.dataset, n=args.n, pool_temp=paths.POOL_TEMP,
                                    pool_k=paths.POOL_K, outputs_dir=args.outputs_dir)
    if args.target == "test":
        out_path = paths.test_embeddings(args.model, args.dataset, n=args.n, pool_temp=paths.POOL_TEMP,
                                         pool_k=paths.POOL_K, variant=args.variant, outputs_dir=args.outputs_dir)
    else:
        out_path = paths.pool_embeddings(args.model, args.dataset, n=args.n, temp=args.pool_temp,
                                         k=args.pool_k, variant=args.variant, outputs_dir=args.outputs_dir)
    if out_path.exists() and not args.force:
        print(f"[skip] exists: {out_path}")
        return

    questions = pd.read_csv(test_csv, usecols=["id", "question"])
    questions["id"] = questions["id"].astype(int)
    if args.target == "test":
        df = pd.read_csv(test_csv)
    else:
        pool = pd.read_pickle(paths.pool_responses(args.model, args.dataset, n=args.n, temp=args.pool_temp,
                                                   k=args.pool_k, outputs_dir=args.outputs_dir))
        pool["id"] = pool["id"].astype(int)
        df = pool.merge(questions, on="id", how="inner").sort_values(["id", "sample_idx"]).reset_index(drop=True)
        if len(df) != len(pool):
            print(f"[warn] {len(pool) - len(df):,} pool rows have no question in {test_csv}")

    texts = [build_text(q, r) for q, r in zip(df["question"].tolist(), df["response"].tolist())]
    print(f"[embed] {len(texts):,} texts with {embed_model}")
    llm = load_vllm_embed(embed_model, args.n_device, enforce_eager=True)
    embeddings = embed_texts(llm, texts)

    os.makedirs(out_path.parent, exist_ok=True)
    if args.target == "test":
        np.savez_compressed(out_path, ids=df["id"].astype(str).to_numpy(), embeddings=embeddings)
    else:
        np.savez_compressed(out_path, ids=df["id"].to_numpy(dtype=np.int64),
                            sample_idx=df["sample_idx"].to_numpy(dtype=np.int32), embeddings=embeddings)
    print(f"[write] {out_path}  shape={embeddings.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="reasoning model whose responses are embedded")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target", choices=("test", "pool"), required=True)
    ap.add_argument("--variant", choices=("own", "gemma"), default="own")
    ap.add_argument("--n", type=int, default=paths.N)
    ap.add_argument("--pool_temp", type=float, default=paths.POOL_TEMP)
    ap.add_argument("--pool_k", type=int, default=paths.POOL_K)
    ap.add_argument("--n_device", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--outputs_dir", default=None)
    main(ap.parse_args())
