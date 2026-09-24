"""BASCData: every cached artifact for one (model, dataset) pair, as arrays.

Loads
  - the probe pool: K sampled responses per question (answers + embeddings),
  - the deployed response per question (T=0.6): answer, correctness, token
    log-probability features and embedding.
Questions whose deployed response has no extractable answer are dropped, so
every method is evaluated on the same questions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import paths


@dataclass
class BASCData:
    dataset: str
    model: str

    ids: np.ndarray                 # (N,) int64, ascending
    answers: np.ndarray             # (N, K) object, pool answers
    embeddings: np.ndarray          # (N, K, D) float32, pool (question, response) embeddings

    test_answers: np.ndarray        # (N,) object, deployed answer
    test_correct: np.ndarray        # (N,) int64
    test_embeddings: np.ndarray     # (N, D') float32
    test_avg_logprobs: np.ndarray   # (N,) mean token log-prob of the full response
    test_ans_logprobs: np.ndarray   # (N,) mean token log-prob of the \boxed{} span

    out_df: pd.DataFrame            # per-question table (question, ground_truth, ...)

    @property
    def n_queries(self) -> int:
        return int(self.ids.shape[0])

    @property
    def k_rollouts(self) -> int:
        return int(self.answers.shape[1])

    @classmethod
    def load(
        cls,
        dataset: str,
        model: str,
        *,
        n: int = paths.N,
        pool_temp: float = paths.POOL_TEMP,
        pool_k: int = paths.POOL_K,
        test_pool_temp: float | None = None,
        test_pool_k: int | None = None,
        variant: str = "own",
        outputs_dir=None,
        verbose: bool = True,
    ) -> "BASCData":
        """`pool_temp`/`pool_k` select the probe pool. The deployed-response files
        are named after a pool stem too; `test_pool_temp`/`test_pool_k` (default:
        the probe pool) let the temperature ablation reuse the canonical
        deployed responses with a different probe pool. `variant` selects the
        embedding backbone ("own" or "gemma")."""
        tt = pool_temp if test_pool_temp is None else test_pool_temp
        tk = pool_k if test_pool_k is None else test_pool_k
        files = {
            "pool_answers": paths.pool_answers(model, dataset, n=n, temp=pool_temp, k=pool_k,
                                               outputs_dir=outputs_dir),
            "pool_embeddings": paths.pool_embeddings(model, dataset, n=n, temp=pool_temp, k=pool_k,
                                                     variant=variant, outputs_dir=outputs_dir),
            "test_responses": paths.test_responses(model, dataset, n=n, pool_temp=tt, pool_k=tk,
                                                   outputs_dir=outputs_dir),
            "test_embeddings": paths.test_embeddings(model, dataset, n=n, pool_temp=tt, pool_k=tk,
                                                     variant=variant, outputs_dir=outputs_dir),
        }
        for key, path in files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"[BASCData] missing {key}: {path}")
        if verbose:
            print(f"[BASCData.load] {model} / {dataset}")

        # --- probe pool answers: rows ordered by (id, sample_idx) -------------
        slim = pd.read_pickle(files["pool_answers"])
        slim["id"] = slim["id"].astype(int)
        slim = slim.sort_values("id", kind="stable").reset_index(drop=True)
        per_id = slim.groupby("id").size()
        if per_id.nunique() != 1:
            raise ValueError(f"Uneven pool size per id: min={per_id.min()} max={per_id.max()}")
        K = int(per_id.iloc[0])
        if K != pool_k:
            raise ValueError(f"pool has k={K} but pool_k={pool_k} was requested")
        ids = np.sort(slim["id"].unique()).astype(np.int64)
        N = len(ids)
        answers = slim["answer"].fillna("").astype(str).to_numpy(dtype=object).reshape(N, K)

        # --- probe pool embeddings, aligned row-wise with the answers ---------
        z = np.load(files["pool_embeddings"], allow_pickle=True)
        emb_ids = np.asarray(z["ids"], dtype=np.int64)
        emb_sidx = np.asarray(z["sample_idx"], dtype=np.int32)
        order = np.lexsort((emb_sidx, emb_ids))
        emb = z["embeddings"].astype(np.float32)[order]
        if not np.array_equal(emb_ids[order], np.repeat(ids, K)):
            raise ValueError("pool embedding ids do not line up with pool answers; regenerate one of them")
        embeddings = emb.reshape(N, K, -1)

        # --- deployed responses -----------------------------------------------
        out_df = pd.read_csv(files["test_responses"])
        out_df["id"] = out_df["id"].astype(int)
        missing = set(ids) - set(out_df["id"].tolist())
        if missing:
            raise ValueError(f"deployed-response table is missing {len(missing)} pool ids")

        # Drop questions whose deployed response has no extractable answer.
        valid = out_df["answer"].notna() & (out_df["answer"].astype(str).str.strip() != "")
        keep = np.intersect1d(np.sort(out_df.loc[valid, "id"].to_numpy()), ids)
        if len(keep) < N and verbose:
            print(f"[BASCData.load] dropping {N - len(keep)} questions with an empty deployed answer")
        mask = np.isin(ids, keep)
        ids, answers, embeddings = ids[mask], answers[mask], embeddings[mask]
        out_df = out_df.set_index("id").loc[ids].reset_index()

        tz = np.load(files["test_embeddings"], allow_pickle=True)
        pos_by_id = {int(i): p for p, i in enumerate(np.asarray(tz["ids"], dtype=np.int64))}
        try:
            test_embeddings = tz["embeddings"].astype(np.float32)[[pos_by_id[int(i)] for i in ids]]
        except KeyError as e:
            raise ValueError(f"deployed-response embeddings are missing id {e.args[0]}")

        data = cls(
            dataset=dataset,
            model=model,
            ids=ids,
            answers=answers,
            embeddings=embeddings,
            test_answers=out_df["answer"].fillna("").astype(str).to_numpy(dtype=object),
            test_correct=out_df["correct"].astype(int).to_numpy(dtype=np.int64),
            test_embeddings=test_embeddings,
            test_avg_logprobs=out_df["avg_logprobs"].astype(float).to_numpy(dtype=np.float32),
            test_ans_logprobs=out_df["ans_logprobs"].astype(float).to_numpy(dtype=np.float32),
            out_df=out_df,
        )
        if verbose:
            print(f"[BASCData.load] N={data.n_queries}, K={K}, D_pool={embeddings.shape[-1]}, "
                  f"D_test={test_embeddings.shape[-1]}, accuracy={data.test_correct.mean():.3f}")
        return data
