"""Answer extraction, correctness against ground truth, and exact-match agreement.

Correctness (answer vs. ground truth):
  - math (GSM8K, PolyMath): exact string match of the extracted \\boxed{} answer.
  - QA (SciQ, TriviaQA, WebQ): case-insensitive, bidirectional substring match
    against any accepted alias, after stripping a leading \\text{ wrapper.

Agreement (answer vs. answer, the "exact" rule used for math and as the
non-semantic rule for QA) mirrors the same two rules applied between two model
answers. Two empty answers agree; one empty answer never agrees.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MATH_DATASETS = frozenset({"gsm8k", "polymath"})
SINGLE_ANSWER_QA = frozenset({"sciq"})
MULTI_ANSWER_QA = frozenset({"trivia_qa", "webq"})
QA_DATASETS = SINGLE_ANSWER_QA | MULTI_ANSWER_QA

# Non-greedy: captures up to the first closing brace of the last \boxed{...}.
BOXED_RE = re.compile(r"\\boxed\s*\{\s*(.*?)\s*\}", flags=re.DOTALL)


def extract_last_boxed(text: str) -> str:
    """Content of the last \\boxed{...} in `text`, or "" if there is none."""
    if not text:
        return ""
    hits = BOXED_RE.findall(text)
    return str(hits[-1]).strip() if hits else ""


def clean_answer(ans, lower: bool = True) -> str:
    a = str(ans).split("\\text{")[-1].replace("\\", "")
    return a.lower() if lower else a


def qa_correct(ans, gts) -> int:
    """1 if the answer and any alias in `gts` contain one another, else 0."""
    if ans is None:
        return 0
    a = str(ans).split("\\text{")[-1].lower().replace("\\", "")
    # An empty answer is a substring of every alias; it is never correct.
    if not a.strip():
        return 0
    for gt in gts:
        t = str(gt).lower()
        if (a in t) or (t in a):
            return 1
    return 0


def compute_correctness(df: pd.DataFrame, dataset: str) -> np.ndarray:
    """Correctness of df["answer"] against df["ground_truth"] (list of aliases
    for multi-answer QA, a single string otherwise)."""
    if dataset in MATH_DATASETS:
        return (df["answer"].fillna("") == df["ground_truth"].fillna("")).astype(int).to_numpy(dtype=np.int64)
    if dataset in MULTI_ANSWER_QA:
        gts_iter = df["ground_truth"]
    elif dataset in SINGLE_ANSWER_QA:
        gts_iter = ([gt] for gt in df["ground_truth"])
    else:
        raise ValueError(f"Unknown dataset for correctness: {dataset}")
    return np.fromiter((qa_correct(a, gts) for a, gts in zip(df["answer"], gts_iter)),
                       dtype=np.int64, count=len(df))


# ---------------------------------------------------------------------------
# Exact-match agreement between two model answers
# ---------------------------------------------------------------------------

def _norm_math(s) -> str:
    return "" if s is None else str(s).strip()


def _norm_qa(s) -> str:
    return "" if s is None else str(s).split("\\text{")[-1].lower().replace("\\", "").strip()


def probe_agree_matrix(anchor_answers: np.ndarray, probe_answers: np.ndarray, dataset: str) -> np.ndarray:
    """(n, m) bool matrix: does each anchor answer agree with each of its m probes?"""
    if anchor_answers.ndim != 1:
        raise ValueError(f"anchor_answers must be 1D, got shape {anchor_answers.shape}")
    if probe_answers.ndim != 2 or probe_answers.shape[0] != anchor_answers.shape[0]:
        raise ValueError(f"probe_answers must be (n, m) with n={anchor_answers.shape[0]}, "
                         f"got {probe_answers.shape}")
    n, m = probe_answers.shape
    out = np.zeros((n, m), dtype=bool)

    if dataset in MATH_DATASETS:
        for i in range(n):
            a = _norm_math(anchor_answers[i])
            for j in range(m):
                out[i, j] = _norm_math(probe_answers[i, j]) == a
        return out

    if dataset in QA_DATASETS:
        for i in range(n):
            a = _norm_qa(anchor_answers[i])
            for j in range(m):
                b = _norm_qa(probe_answers[i, j])
                if not a and not b:
                    out[i, j] = True
                elif not a or not b:
                    out[i, j] = False
                else:
                    out[i, j] = (a in b) or (b in a)
        return out

    raise ValueError(f"Unknown dataset for agreement rule: {dataset}")
