"""Evaluation metrics.

ECE1 / ECE2 are the L1 / L2 top-label calibration errors of Kumar et al. (2019)
(`uncertainty-calibration` package) with 12 equal-mass bins. Brier score and
AUROC come from scikit-learn.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

import calibration as cal
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

N_BINS = 12


def get_ece(conf, labels, n_bins: int = N_BINS, p: int = 2) -> float:
    return cal.lower_bound_scaling_ce(conf, labels, p=p, debias=False, num_bins=n_bins,
                                      binning_scheme=cal.get_equal_bins, mode="top-label")


def get_ece1(conf, labels) -> float:
    return get_ece(conf, labels, n_bins=N_BINS, p=1)


def get_ece2(conf, labels) -> float:
    return get_ece(conf, labels, n_bins=N_BINS, p=2)


def get_nll(y_pred, y_true) -> float:
    y_pred = np.clip(y_pred, 1e-6, 1.0 - 1e-6)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def get_mce(y_pred, y_true, n_bins: int = N_BINS) -> float:
    """Maximum calibration error over equal-mass bins."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="quantile")
    return float(np.max(np.abs(prob_true - prob_pred)))


def correctness_metrics(preds: np.ndarray, correct: np.ndarray) -> Dict[str, float]:
    preds = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)
    correct = np.asarray(correct, dtype=int)
    return {
        "ECE1": get_ece1(preds, correct),
        "ECE2": get_ece2(preds, correct),
        "MCE": get_mce(preds, correct),
        "NLL": get_nll(preds, correct),
        "Brier": brier_score_loss(correct, preds),
        "AUROC": roc_auc_score(correct, preds) if len(np.unique(correct)) == 2 else float("nan"),
    }


def agreement_metrics(preds: np.ndarray, agree: np.ndarray) -> Dict[str, float]:
    """Calibration against held-out binary probe agreement (the quantity the
    binary-isotonic guarantee bounds)."""
    preds = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)
    agree = np.asarray(agree, dtype=int)
    return {
        "agree_ECE1": get_ece1(preds, agree),
        "agree_ECE2": get_ece2(preds, agree),
        "agree_MCE": get_mce(preds, agree),
        "agree_Brier": brier_score_loss(agree, preds),
        "agree_AUROC": roc_auc_score(agree, preds) if len(np.unique(agree)) == 2 else float("nan"),
    }


def ece_decomposition(preds: np.ndarray, correct: np.ndarray, agree: np.ndarray,
                      n_bins: int = N_BINS) -> Dict[str, float]:
    """Binned estimates of the terms in ECE_Z <= ECE_S + ProxyGap (Section 3),
    using a shared equal-width binning of the predictions."""
    preds = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(preds, edges[1:-1], right=False), 0, n_bins - 1)
    populations = np.bincount(bins, minlength=n_bins).astype(np.int64)
    conf_bin, z_bin, s_bin = (np.zeros(n_bins) for _ in range(3))
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            conf_bin[b] = preds[mask].mean()
            z_bin[b] = np.asarray(correct, dtype=float)[mask].mean()
            s_bin[b] = np.asarray(agree, dtype=float)[mask].mean()
    w = populations / populations.sum()
    return {
        "ECE_Z": float(np.sum(w * np.abs(z_bin - conf_bin))),
        "ECE_S": float(np.sum(w * np.abs(s_bin - conf_bin))),
        "ProxyGap": float(np.sum(w * np.abs(z_bin - s_bin))),
    }


def bundle_all_metrics(preds: np.ndarray, correct: np.ndarray, agree: Optional[np.ndarray]) -> Dict[str, float]:
    """Correctness metrics, plus agreement metrics and the ECE decomposition
    when a held-out test probe `agree` is available."""
    out = correctness_metrics(preds, correct)
    if agree is not None:
        out.update(agreement_metrics(preds, agree))
        out.update(ece_decomposition(preds, correct, agree))
    return out


def reliability_bins(scores: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> Dict[str, str]:
    """Equal-width reliability bins on [0, 1] as JSON strings (count, mean
    prediction, mean correctness per bin; None for empty bins) so that bins can
    be pooled across seeds, models and datasets for reliability diagrams."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    mean_p: List = [None] * n_bins
    mean_c: List = [None] * n_bins
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b]:
            mean_p[b] = float(scores[mask].mean())
            mean_c[b] = float(correct[mask].mean())
    return {
        "rel_counts": json.dumps(counts.tolist()),
        "rel_pred": json.dumps(mean_p),
        "rel_correct": json.dumps(mean_c),
    }


def selective_acc_at_abstention(scores: np.ndarray, correct: np.ndarray,
                                abstention_rates: Iterable[float]) -> pd.DataFrame:
    """Abstain on the lowest-confidence fraction `a` of examples, for each rate
    `a`, and report accuracy / mean confidence of the kept and rejected sets.

    Columns: rate, sel_acc, sel_conf, rej_conf, rej_acc.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    idx = np.argsort(-scores)
    correct_sorted, scores_sorted = correct[idx], scores[idx]
    n = len(correct_sorted)
    rows = []
    for a in abstention_rates:
        k = int(np.floor((1.0 - a) * n))
        if k <= 0:
            rows.append([a, np.nan, np.nan, np.nan, np.nan])
            continue
        sa = float(np.mean(correct_sorted[:k]))
        if a == 0.0:
            sc, rc, ra = 0.0, 0.0, 0.0
        else:
            sc = float(np.mean(scores_sorted[:k]))
            rc = float(np.mean(scores_sorted[k:]))
            ra = float(np.mean(correct_sorted[k:]))
        rows.append([a, sa, sc, rc, ra])
    return pd.DataFrame(rows, columns=["rate", "sel_acc", "sel_conf", "rej_conf", "rej_acc"])
