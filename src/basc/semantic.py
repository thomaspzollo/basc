"""Semantic agreement for QA via bidirectional entailment (Kuhn et al., 2023).

Two answers to the same question agree iff an NLI cross-encoder labels both
  "Question: {q}\\nAnswer: {a}"  ->  "Question: {q}\\nAnswer: {b}"
and the reverse direction as entailment. Empty-answer handling matches the
exact rule in `basc.answers`: two empty answers agree, one empty never does.

NLI is expensive, so the (N, K) matrix of agreement between each question's
deployed answer and every pool response is computed once per (model, dataset)
by scripts/pipeline/semantic_agreement.py and cached to disk.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, Optional

import numpy as np

from . import paths
from .answers import QA_DATASETS

NLI_MODEL_NAME_DEFAULT = "cross-encoder/nli-deberta-v3-base"
_LABEL_ENTAILMENT = 1   # CrossEncoder logits are ordered [contradiction, entailment, neutral]


def _clean_answer(x) -> str:
    """Strip \\text{...} / \\boxed{...} wrappers."""
    if x is None:
        return ""
    x = str(x).strip()
    if not x:
        return ""
    for wrapper in ("text", "boxed"):
        m = re.fullmatch(rf"\\{wrapper}\{{(.*)\}}", x, flags=re.DOTALL)
        if m:
            x = m.group(1).strip()
    for prefix in (r"\text{", r"\boxed{"):
        if x.startswith(prefix) and not x.endswith("}"):
            x = x[len(prefix):].strip()
    return x.strip()


def _qa_context(question, answer) -> str:
    a = _clean_answer(answer)
    q = "" if question is None else str(question).strip()
    return f"Question: {q}\nAnswer: {a}" if a else ""


def build_anchor_mutual_entailment(
    questions: Iterable,
    anchor_answers: np.ndarray,
    pool_answers: np.ndarray,
    *,
    nli_model_name: str = NLI_MODEL_NAME_DEFAULT,
    batch_size: int = 1024,
    chunk_pairs: int = 200_000,
    device: Optional[str] = None,
    fp16: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """(N, K) bool matrix; entry [q, j] is True iff question q's anchor answer and
    pool response j mutually entail each other."""
    from sentence_transformers import CrossEncoder
    from tqdm.auto import tqdm

    questions = list(questions)
    anchors = np.asarray(anchor_answers, dtype=object)
    pool = np.asarray(pool_answers, dtype=object)
    if pool.ndim != 2 or anchors.shape != (pool.shape[0],):
        raise ValueError(f"need anchor_answers (N,) and pool_answers (N, K); got "
                         f"{anchors.shape} and {pool.shape}")
    N, K = pool.shape
    if len(questions) != N:
        raise ValueError(f"questions length {len(questions)} != N={N}")

    anchor_ctx = np.array([_qa_context(questions[i], anchors[i]) for i in range(N)], dtype=object)
    pool_ctx = np.array([[_qa_context(questions[i], pool[i, j]) for j in range(K)] for i in range(N)],
                        dtype=object)
    anchor_ne = anchor_ctx != ""
    pool_ne = pool_ctx != ""

    pair_q, pair_j = np.nonzero(anchor_ne[:, None] & pool_ne)
    n_pairs = int(pair_q.shape[0])

    # Row 0: anchor -> pool entailment; row 1: pool -> anchor entailment.
    entail = np.zeros((2, n_pairs), dtype=bool)
    if n_pairs > 0:
        model = CrossEncoder(nli_model_name, device=device)
        if fp16:
            model.model.half()
        starts = range(0, n_pairs, chunk_pairs)
        if show_progress:
            starts = tqdm(starts, total=(n_pairs + chunk_pairs - 1) // chunk_pairs, desc="NLI chunks")
        for start in starts:
            end = min(start + chunk_pairs, n_pairs)
            qs, js = pair_q[start:end], pair_j[start:end]
            fwd = [(anchor_ctx[q], pool_ctx[q, j]) for q, j in zip(qs, js)]
            bwd = [(pool_ctx[q, j], anchor_ctx[q]) for q, j in zip(qs, js)]
            scores = np.asarray(model.predict(fwd + bwd, batch_size=batch_size, show_progress_bar=False))
            labels = scores.argmax(axis=1) == _LABEL_ENTAILMENT
            entail[0, start:end] = labels[: end - start]
            entail[1, start:end] = labels[end - start:]

    out = np.zeros((N, K), dtype=bool)
    out[pair_q, pair_j] = entail[0] & entail[1]
    out |= (~anchor_ne)[:, None] & ~pool_ne
    return out


def save_semantic_cache(path, *, ids: np.ndarray, semantic_agree: np.ndarray, nli_model_name: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=np.int64),
        semantic_agree=np.asarray(semantic_agree, dtype=bool),
        nli_model_name=np.array(nli_model_name),
    )


def load_semantic_cache(path, *, expected_ids: np.ndarray, expected_shape: tuple) -> np.ndarray:
    """Load a cached (N, K) matrix, reindexed to `expected_ids` (the cache may
    cover a superset of the ids kept by BASCData)."""
    z = np.load(path, allow_pickle=True)
    mat = np.asarray(z["semantic_agree"], dtype=bool)
    cached_ids = np.asarray(z["ids"], dtype=np.int64)
    exp_ids = np.asarray(expected_ids, dtype=np.int64)
    if not np.array_equal(cached_ids, exp_ids):
        pos_by_id = {int(i): p for p, i in enumerate(cached_ids)}
        missing = [int(i) for i in exp_ids if int(i) not in pos_by_id]
        if missing:
            raise ValueError(f"semantic cache is missing {len(missing)} ids (e.g. {missing[:5]}); "
                             f"rebuild {path}")
        mat = mat[np.array([pos_by_id[int(i)] for i in exp_ids], dtype=np.int64)]
    if mat.shape != expected_shape:
        raise ValueError(f"semantic cache shape {mat.shape} != expected {expected_shape} ({path})")
    return mat


def load_anchor_semantic(data, *, n: int = paths.N, pool_temp: float = paths.POOL_TEMP,
                         pool_k: int = paths.POOL_K, outputs_dir=None,
                         verbose: bool = True) -> Optional[np.ndarray]:
    """(N, K) semantic agreement for a BASCData bundle, aligned to `data.ids`.
    None for math datasets or when the cache has not been built (callers then
    fall back to exact agreement)."""
    if data.dataset not in QA_DATASETS:
        return None
    path = paths.anchor_semantic_agreement(data.model, data.dataset, n=n, temp=pool_temp, k=pool_k,
                                           outputs_dir=outputs_dir)
    if not os.path.exists(path):
        if verbose:
            print(f"[semantic-skip] no cache at {path}")
        return None
    try:
        return load_semantic_cache(path, expected_ids=data.ids,
                                   expected_shape=(data.n_queries, data.k_rollouts))
    except ValueError as e:
        if verbose:
            print(f"[semantic-skip] {path}: {e}")
        return None
