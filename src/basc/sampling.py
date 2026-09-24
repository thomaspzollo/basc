"""Calibration/test splits and anchor + probe draws.

Every calibration question contributes one *anchor* response -- the deployed
T=0.6 answer, i.e. the same kind of response BASC is evaluated on -- plus some
number of *probe* responses drawn without replacement from its T=0.7 pool.
Binary agreement B = 1[probe answer == anchor answer] is the training signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .answers import probe_agree_matrix
from .data import BASCData


def three_way_split(
    n_queries: int,
    *,
    seed: int,
    score_frac: float = 0.20,
    iso_frac: float = 0.20,
    test_frac: float = 0.60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split positions [0, n_queries) into (D_score, D_iso, D_test): 20/20/60 by
    default, i.e. a 40/60 calibration/test split whose calibration half is
    divided evenly between Stage 1 and Stage 2."""
    if not np.isclose(score_frac + iso_frac + test_frac, 1.0, atol=1e-8):
        raise ValueError("split fractions must sum to 1")
    positions = np.arange(n_queries, dtype=np.int64)
    cal_pos, test_pos = train_test_split(positions, test_size=test_frac, random_state=seed, shuffle=True)
    score_pos, iso_pos = train_test_split(
        cal_pos, test_size=iso_frac / (score_frac + iso_frac), random_state=seed, shuffle=True,
    )
    return (np.sort(score_pos).astype(np.int64),
            np.sort(iso_pos).astype(np.int64),
            np.sort(test_pos).astype(np.int64))


def draw_probe_indices(n: int, k_rollouts: int, n_probes: int, *, seed: int) -> np.ndarray:
    """(n, n_probes) pool indices, drawn without replacement per question.

    Each question draws 1 + n_probes distinct indices and the first is
    discarded: earlier versions of the code used it as a pool anchor, and
    keeping the draw preserves the exact random streams behind the paper's
    numbers.
    """
    if n_probes + 1 > k_rollouts:
        raise ValueError(f"need {n_probes} probes but the pool has only {k_rollouts} responses")
    rng = np.random.default_rng(seed)
    probes = np.empty((n, n_probes), dtype=np.int32)
    for i in range(n):
        probes[i] = rng.choice(k_rollouts, size=n_probes + 1, replace=False)[1:]
    return probes


@dataclass(frozen=True)
class AnchorSet:
    """Anchors and probes for one split.

    query_pos  (n,)    positions into BASCData arrays
    X          (n, D)  anchor embeddings
    answers    (n,)    anchor answers
    correct    (n,)    anchor correctness (used only by the supervised reference)
    probe_sidx (n, m)  pool indices of the probes
    agree      (n, m)  exact-rule anchor/probe agreement
    sem_agree  (n, m)  semantic agreement, or None if no semantic cache was given
    """
    query_pos: np.ndarray
    X: np.ndarray
    answers: np.ndarray
    correct: np.ndarray
    probe_sidx: np.ndarray
    agree: np.ndarray
    sem_agree: Optional[np.ndarray]

    @property
    def n(self) -> int:
        return int(self.query_pos.shape[0])

    @property
    def agreement(self) -> np.ndarray:
        """The paper's agreement rule: semantic when available (QA), else exact."""
        return self.sem_agree if self.sem_agree is not None else self.agree


def draw_anchor_set(
    data: BASCData,
    query_pos: np.ndarray,
    n_probes: int,
    *,
    seed: int,
    anchor_semantic: Optional[np.ndarray] = None,
) -> AnchorSet:
    """Anchor = each question's deployed answer; `n_probes` probes from its pool.
    `anchor_semantic` is the (N, K) semantic cache from `load_anchor_semantic`."""
    qp = query_pos
    probe_sidx = draw_probe_indices(len(qp), data.k_rollouts, n_probes, seed=seed)
    answers = data.test_answers[qp]
    probe_answers = data.answers[qp[:, None], probe_sidx]
    agree = probe_agree_matrix(np.asarray(answers, dtype=object),
                               np.asarray(probe_answers, dtype=object), data.dataset)
    sem = anchor_semantic[qp[:, None], probe_sidx].astype(bool) if anchor_semantic is not None else None
    return AnchorSet(query_pos=qp, X=data.test_embeddings[qp], answers=answers,
                     correct=data.test_correct[qp].astype(np.int64), probe_sidx=probe_sidx,
                     agree=agree, sem_agree=sem)


def empirical_sc(agree: np.ndarray) -> np.ndarray:
    """Per-question self-consistency estimate: mean agreement over the m probes."""
    agree = np.asarray(agree, dtype=float)
    if agree.ndim != 2 or agree.shape[1] < 1:
        raise ValueError(f"agree must be (n, m>=1); got {agree.shape}")
    return agree.mean(axis=1)
