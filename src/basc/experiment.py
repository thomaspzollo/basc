"""One trial of every calibration-set method on a shared split.

`run_id_trial` (in-distribution) and `run_shift_trial` (distribution shift)
draw the splits, anchors and probes once and fit every method on them, so all
methods are compared on identical data. Both return per-test-question
confidence scores; metrics are computed by the caller.

Method keys:
  basc_k{k}              BASC-L: binomial-logistic Stage 1 on D_score with k-1
                         probes per question (k decodes incl. the anchor),
                         binary isotonic Stage 2 on D_iso with one probe each.
  basc_sem_k{k}          same with semantic (NLI) agreement (QA only).
  ridge_clip_sc[_sem]_k{k}  RR-SC: ridge on the empirical SC rate from k-1
                         probes, trained on D_score + D_iso (one stage).
  ssc_ridge_k{k}         RR-SSC: ridge on the SSC target of the first k pool
                         responses of every calibration question.
  sup_logreg_iso_k1      supervised two-stage reference (correctness labels).
  oracle_sc_k100         TT-SC: agreement of the deployed answer with all
                         100 pool responses (needs 100 test-time samples).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .answers import probe_agree_matrix
from .baselines import compute_ssc_matrix, fit_predict_ridge_clip, fit_predict_supervised_two_stage
from .data import BASCData
from .sampling import AnchorSet, draw_anchor_set, empirical_sc, three_way_split
from .stage1 import fit_predict_binomial_logistic
from .stage2 import fit_binary_isotonic
from .verbal_confidence import VERBAL_CONF_STEM_IDX, impute_missing_with_mean, load_verbal_conf_scores


def fit_all_methods(
    *,
    score: AnchorSet,
    iso: AnchorSet,
    lr_iso: AnchorSet,
    X_test: np.ndarray,
    seed: int,
    max_k: int,
    k_variants: Iterable[int],
    ridge_k_variants: Iterable[int],
    ssc_k_variants: Iterable[int],
    ssc_c_mat: Optional[np.ndarray],
    ssc_answers: Optional[np.ndarray],
    ssc_embeddings: np.ndarray,
    ssc_train_pos: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Fit every calibration-set method; return {method: test-set scores}.

    score  -- D_score anchors with max_k-1 probes each (BASC Stage 1).
    iso    -- D_iso anchors with one probe each (BASC Stage 2).
    lr_iso -- D_iso anchors with max_k-1 independently drawn probes, so the
              one-stage RR-SC baseline trains on D_score + D_iso at k probes.
    """
    scores: Dict[str, np.ndarray] = {}
    X_iso_test = np.concatenate([iso.X, X_test], axis=0)

    def _basc(agree_score, agree_iso, prefix):
        B_iso = agree_iso[:, 0].astype(np.int64)
        for k in k_variants:
            if k < 2 or k > max_k:
                continue
            m = np.full(score.n, k - 1, dtype=np.int64)
            c = agree_score[:, : k - 1].sum(axis=1).astype(np.int64)
            raw = fit_predict_binomial_logistic(score.X, m, c, X_iso_test)
            scores[f"{prefix}_k{k}"] = fit_binary_isotonic(raw[: iso.n], B_iso).predict(raw[iso.n:])

    _basc(score.agree, iso.agree, "basc")
    if score.sem_agree is not None and iso.sem_agree is not None:
        _basc(score.sem_agree, iso.sem_agree, "basc_sem")

    # RR-SC on D_score + D_iso.
    X_lr = np.concatenate([score.X, lr_iso.X], axis=0)
    lr_agree = {"": np.concatenate([score.agree, lr_iso.agree], axis=0)}
    if score.sem_agree is not None and lr_iso.sem_agree is not None:
        lr_agree["_sem"] = np.concatenate([score.sem_agree, lr_iso.sem_agree], axis=0)
    for k in ridge_k_variants:
        if k < 2 or k > max_k:
            continue
        for tag, agree in lr_agree.items():
            y_sc = empirical_sc(agree[:, : k - 1])
            scores[f"ridge_clip_sc{tag}_k{k}"] = fit_predict_ridge_clip(X_lr, y_sc, X_test, random_state=seed)

    # RR-SSC: rows are the first k pool responses of every calibration
    # question, target = their SSC; no Stage 2 (Huang et al.'s recipe).
    if ssc_c_mat is not None and ssc_answers is not None:
        n_train = len(ssc_train_pos)
        for k in ssc_k_variants:
            if k < 2 or k > ssc_c_mat.shape[1] or k > max_k:
                continue
            ssc_mat = compute_ssc_matrix(ssc_answers, ssc_c_mat, k)
            qp_exp = np.repeat(ssc_train_pos, k)
            sidx_exp = np.tile(np.arange(k), n_train)
            X_train = ssc_embeddings[qp_exp, sidx_exp]
            y_train = ssc_mat[qp_exp, sidx_exp]
            valid = np.isfinite(y_train)
            if valid.any():
                scores[f"ssc_ridge_k{k}"] = fit_predict_ridge_clip(X_train[valid], y_train[valid], X_test,
                                                                   random_state=seed)

    # Supervised two-stage reference: same anchors, correctness labels.
    scores["sup_logreg_iso_k1"] = fit_predict_supervised_two_stage(
        score.X, score.correct, iso.X, iso.correct, X_test, random_state=seed,
    )
    return scores


def oracle_sc(test: AnchorSet, data: BASCData) -> np.ndarray:
    """TT-SC: mean agreement of each deployed answer with all K pool responses."""
    return probe_agree_matrix(np.asarray(test.answers, dtype=object),
                              data.answers[test.query_pos], data.dataset).mean(axis=1).astype(float)


def run_id_trial(
    data: BASCData,
    seed: int,
    *,
    k_variants: Iterable[int] = (2, 5, 100),
    ridge_k_variants: Iterable[int] = (5, 100),
    ssc_k_variants: Iterable[int] = (),
    ssc_c_mat: Optional[np.ndarray] = None,
    ssc_answers: Optional[np.ndarray] = None,
    anchor_semantic: Optional[np.ndarray] = None,
):
    """In-distribution trial on a 20/20/60 D_score/D_iso/D_test split.

    Returns (scores, correct_test, agree_test, test_ids, sem_agree_test):
      scores          {method: (n_test,) confidence}
      correct_test    correctness of the deployed test answers
      agree_test      exact agreement of each test answer with one held-out probe
      test_ids        question ids of D_test (to align external baselines)
      sem_agree_test  semantic version of agree_test, or None
    """
    max_k = data.k_rollouts
    score_pos, iso_pos, test_pos = three_way_split(data.n_queries, seed=seed)
    score = draw_anchor_set(data, score_pos, max_k - 1, seed=seed * 3 + 1, anchor_semantic=anchor_semantic)
    iso = draw_anchor_set(data, iso_pos, 1, seed=seed * 3 + 2, anchor_semantic=anchor_semantic)
    lr_iso = draw_anchor_set(data, iso_pos, max_k - 1, seed=seed * 3 + 4, anchor_semantic=anchor_semantic)
    test = draw_anchor_set(data, test_pos, max_k - 1, seed=seed * 3 + 3, anchor_semantic=anchor_semantic)

    scores = fit_all_methods(
        score=score, iso=iso, lr_iso=lr_iso, X_test=test.X, seed=seed, max_k=max_k,
        k_variants=k_variants, ridge_k_variants=ridge_k_variants,
        ssc_k_variants=ssc_k_variants, ssc_c_mat=ssc_c_mat, ssc_answers=ssc_answers,
        ssc_embeddings=data.embeddings, ssc_train_pos=np.concatenate([score.query_pos, iso.query_pos]),
    )
    scores["oracle_sc_k100"] = oracle_sc(test, data)
    sem_agree_test = test.sem_agree[:, 0].astype(np.int64) if test.sem_agree is not None else None
    return scores, test.correct, test.agree[:, 0].astype(np.int64), data.ids[test.query_pos], sem_agree_test


def external_baselines(data: BASCData, test_ids: np.ndarray, *, vc_path,
                       float64_logprobs: bool = False):
    """Single-generation baselines on the deployed responses of `test_ids`:
    TP (geometric-mean token probability of the full response), ATP (of the
    \\boxed{} answer tokens) and VC (verbalized confidence, prompt stem 6;
    unparseable outputs imputed with the mean).

    Returns ({method: scores in [0, 1]}, correct). The main sweep reads the
    float32 log-probabilities held by BASCData; the selection and shift sweeps
    read the float64 CSV columns (`float64_logprobs=True`). Both are kept so
    reruns match the paper's result files exactly.
    """
    id_to_pos = {int(i): p for p, i in enumerate(data.ids)}
    pos = np.array([id_to_pos[int(i)] for i in test_ids], dtype=np.int64)
    if float64_logprobs:
        avg_lp = data.out_df["avg_logprobs"].to_numpy(dtype=float)[pos]
        ans_lp = data.out_df["ans_logprobs"].to_numpy(dtype=float)[pos]
    else:
        avg_lp = data.test_avg_logprobs[pos].astype(float)
        ans_lp = data.test_ans_logprobs[pos].astype(float)
    scores = {"logprob": np.clip(np.exp(avg_lp), 0.0, 1.0),
              "ans_logprob": np.clip(np.exp(ans_lp), 0.0, 1.0)}
    vc = load_verbal_conf_scores(vc_path, pd.Series(np.asarray(test_ids).astype(int)))
    if vc is not None and np.isfinite(vc).any():
        scores[f"verbal_conf_stem{VERBAL_CONF_STEM_IDX}"] = np.clip(impute_missing_with_mean(vc), 0.0, 1.0)
    return scores, data.test_correct[pos].astype(int)


def positions_for_ids(data: BASCData, ids: Optional[np.ndarray]) -> np.ndarray:
    """Positions of `ids` in `data` (all positions if None). Ids dropped by
    BASCData (empty deployed answer) are skipped."""
    if ids is None:
        return np.arange(data.n_queries, dtype=np.int64)
    ids_arr = np.asarray(ids, dtype=data.ids.dtype)
    pos = np.clip(np.searchsorted(data.ids, ids_arr), 0, len(data.ids) - 1)
    return pos[data.ids[pos] == ids_arr].astype(np.int64)


def split_train_for_shift(train_pos: np.ndarray, seed: int, iso_frac: float = 0.5):
    """Split the training-domain questions into (D_score, D_iso), 50/50."""
    perm = np.random.default_rng(seed).permutation(train_pos)
    n_iso = int(np.floor(len(perm) * iso_frac))
    return np.sort(perm[n_iso:]), np.sort(perm[:n_iso])


def run_shift_trial(
    train_data: BASCData,
    test_data: BASCData,
    *,
    seed: int,
    train_id_subset: Optional[np.ndarray] = None,
    test_id_subset: Optional[np.ndarray] = None,
    k_variants: Iterable[int] = (2, 5, 100),
    ridge_k_variants: Iterable[int] = (5, 100),
    ssc_k_variants: Iterable[int] = (),
    ssc_c_mat_train: Optional[np.ndarray] = None,
    ssc_answers_train: Optional[np.ndarray] = None,
    anchor_semantic_train: Optional[np.ndarray] = None,
    anchor_semantic_test: Optional[np.ndarray] = None,
):
    """Distribution-shift trial: every method is calibrated on the training
    domain (split 50/50 into D_score / D_iso) and evaluated on the test domain.
    `*_id_subset` restrict either side to a subset of question ids (e.g.
    PolyMath language blocks). Returns (scores, correct, agree, test_ids)."""
    train_pos = positions_for_ids(train_data, train_id_subset)
    test_pos = positions_for_ids(test_data, test_id_subset)
    score_pos, iso_pos = split_train_for_shift(train_pos, seed)

    max_k = train_data.k_rollouts
    sem = dict(anchor_semantic=anchor_semantic_train)
    score = draw_anchor_set(train_data, score_pos, max_k - 1, seed=seed * 3 + 1, **sem)
    iso = draw_anchor_set(train_data, iso_pos, 1, seed=seed * 3 + 2, **sem)
    lr_iso = draw_anchor_set(train_data, iso_pos, max_k - 1, seed=seed * 3 + 4, **sem)
    test = draw_anchor_set(test_data, test_pos, 1, seed=seed * 3 + 3, anchor_semantic=anchor_semantic_test)

    scores = fit_all_methods(
        score=score, iso=iso, lr_iso=lr_iso, X_test=test.X, seed=seed, max_k=max_k,
        k_variants=k_variants, ridge_k_variants=ridge_k_variants,
        ssc_k_variants=ssc_k_variants, ssc_c_mat=ssc_c_mat_train, ssc_answers=ssc_answers_train,
        ssc_embeddings=train_data.embeddings, ssc_train_pos=np.concatenate([score_pos, iso_pos]),
    )
    scores["oracle_sc_k100"] = oracle_sc(test, test_data)
    return scores, test.correct, test.agree[:, 0].astype(np.int64), test_data.ids[test.query_pos]
