"""Embedding-based baselines and references.

  RR-SC   ridge regression from anchor embeddings to the empirical
          self-consistency rate (one stage, clipped to [0, 1]).
  RR-SSC  the same regressor fit to the confidence-weighted soft
          self-consistency (SSC) target of Huang et al. (2025).
  Sup.    the two-stage BASC recipe with true correctness labels in place of
          agreement (logistic regression on D_score, binary isotonic on D_iso).
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import paths
from .answers import clean_answer
from .stage2 import fit_binary_isotonic


def fit_predict_ridge_clip(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                           *, random_state: int = 42) -> np.ndarray:
    """Standardize + Ridge(alpha=1) on a [0, 1] target; predictions clipped to [0, 1]."""
    y = np.clip(np.asarray(y_train, dtype=float), 0.0, 1.0)
    ridge = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=random_state))])
    ridge.fit(X_train, y)
    return np.clip(ridge.predict(X_test), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Soft self-consistency target (Huang et al., 2025)
# ---------------------------------------------------------------------------

# ROLLOUTS_ssc_k{K}.csv files are searched largest-K first.
_SSC_K_CANDIDATES: Tuple[int, ...] = (10, 20, 32, 100)


def load_ssc_c_matrix(model: str, dataset: str, ids: np.ndarray, *, n: int = paths.N,
                      pool_temp: float = paths.POOL_TEMP, pool_k: int = paths.POOL_K,
                      outputs_dir=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Per-pool-response P(Yes) scores c and answers as (N, K_ssc) matrices
    aligned with `ids`, from scripts/baselines/ssc_scores.py. Returns
    (None, None) when no SSC file exists for (model, dataset)."""
    for K in sorted(_SSC_K_CANDIDATES, reverse=True):
        path = paths.ssc_scores(model, dataset, n=n, temp=pool_temp, k=pool_k, k_ssc=K,
                                outputs_dir=outputs_dir)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        id_to_pos = {int(i): p for p, i in enumerate(ids)}
        c_mat = np.full((len(ids), K), np.nan, dtype=float)
        answers = np.full((len(ids), K), "", dtype=object)
        for row in df.itertuples():
            qp = id_to_pos.get(int(row.id))
            sidx = int(row.sample_idx)
            if qp is None or sidx >= K:
                continue
            c_mat[qp, sidx] = float(row.c)
            answers[qp, sidx] = "" if pd.isna(row.answer) else str(row.answer)
        return c_mat, answers
    return None, None


def compute_ssc_matrix(answers: np.ndarray, c_mat: np.ndarray, k: int) -> np.ndarray:
    """SSC of each of the first k pool responses per question:

        SSC(y_i) = sum_{j: a_j == a_i} c_j / sum_j c_j

    with answers grouped by exact match of `clean_answer`. Rows whose c values
    sum to zero are left NaN."""
    N = answers.shape[0]
    out = np.full((N, k), np.nan, dtype=float)
    for qp in range(N):
        c_row, a_row = c_mat[qp, :k], answers[qp, :k]
        denom = float(np.nansum(c_row))
        if denom <= 0 or not np.isfinite(denom):
            continue
        keys: dict[str, int] = {}
        cids = np.zeros(k, dtype=int)
        for i, a in enumerate(a_row):
            cids[i] = keys.setdefault(clean_answer(a or "", lower=True), len(keys))
        mass = defaultdict(float)
        for cid, cv in zip(cids, c_row):
            if np.isfinite(cv):
                mass[int(cid)] += float(cv)
        out[qp, :] = np.array([mass[int(cid)] / denom for cid in cids])
    return out


# ---------------------------------------------------------------------------
# Supervised reference (uses correctness labels)
# ---------------------------------------------------------------------------

def fit_predict_supervised_two_stage(X_score, y_score, X_iso, y_iso, X_test, *,
                                     C: float = 1.0, random_state: int = 42,
                                     max_iter: int = 5000) -> np.ndarray:
    """Logistic regression on D_score -> correctness, then BASC's binary
    isotonic Stage 2 on D_iso -> correctness."""
    y_score = np.asarray(y_score, dtype=int)
    if len(np.unique(y_score)) < 2:
        rate = float(y_score.mean() if len(y_score) else 0.5)
        return np.full(X_test.shape[0], rate, dtype=float)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, solver="lbfgs")),
    ])
    pipe.fit(X_score, y_score)
    raw_iso = pipe.predict_proba(X_iso)[:, 1]
    raw_test = pipe.predict_proba(X_test)[:, 1]
    y_iso = np.asarray(y_iso, dtype=int)
    if len(np.unique(y_iso)) < 2:
        return np.clip(raw_test, 0.0, 1.0)
    return fit_binary_isotonic(raw_iso, y_iso.astype(np.int64)).predict(raw_test)
