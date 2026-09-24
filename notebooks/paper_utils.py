"""Shared helpers for the paper notebooks: file locations, method colors/labels,
the QA semantic-agreement rule, and pooled reliability bins."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from basc.paths import FIGURES_DIR, RESULTS_DIR  # noqa: F401  (re-exported)

TABLES_DIR = FIGURES_DIR / "tables"

# Calibration error reported in the main-text tables (Tables 1-2), the
# shift figures and Figures 2-3: "ECE1" (as in the paper) or "ECE2".
MAIN_METRIC = os.environ.get("BASC_MAIN_METRIC", "ECE1")
METRIC_TEX = {"ECE1": "ECE$_1$", "ECE2": "ECE$_2$"}
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_MAP = {
    "gsm8k": "GSM8K",
    "polymath": "Polymath",
    "trivia_qa": "Trivia QA",
    "sciq": "SciQ",
    "webq": "WebQ",
}
SEMANTIC_QA = {"sciq", "trivia_qa", "webq"}

METHOD_COLOR = {
    # single-generation baselines
    "logprob":            "#4C72B0",
    "ans_logprob":        "#8172B3",
    "verbal_conf_stem6":  "#55A868",
    # BASC (ours)
    "basc_k2":            "#C44E52",
    "basc_k5":            "#DD8452",
    "basc_k10":           "#E56B41",
    "basc_k20":           "#A83E42",
    "basc_k100":          "#B0553C",
    "basc_bb_k2":         "#937860",
    "basc_bb_k5":         "#DA8BC3",
    "basc_tp":            "#1B9E9E",
    # references
    "oracle_sc_k100":     "#8C8C8C",
    "sup_logreg_iso_k1":  "#17BECF",
    # LR-SC
    "ridge_clip_sc_k2":   "#E4CE94",
    "ridge_clip_sc_k5":   "#CCB974",
    "ridge_clip_sc_k10":  "#A6BB7E",
    "ridge_clip_sc_k20":  "#80BD9A",
    "ridge_clip_sc_k100": "#64B5CD",
    # LR-SSC
    "ssc_ridge_k2":       "#FEEECD",
    "ssc_ridge_k5":       "#FDCC8A",
    "ssc_ridge_k10":      "#DA8BC3",
    "ssc_ridge_k20":      "#B30000",
}

METHOD_LABEL = {
    "logprob":            "TP",
    "ans_logprob":        "ATP",
    "verbal_conf_stem6":  "VC",
    "basc_k2":            "BASC-L (k=2)",
    "basc_k5":            "BASC-L (k=5)",
    "basc_k10":           "BASC-L (k=10)",
    "basc_k20":           "BASC-L (k=20)",
    "basc_k100":          "BASC-L (k=100)",
    "basc_bb_k2":         "BASC-L-bb (k=2)",
    "basc_bb_k5":         "BASC-L-bb (k=5)",
    "basc_tp":            "BASC-TP",
    "oracle_sc_k100":     "TT-SC",
    "sup_logreg_iso_k1":  "Sup. (2-stage)",
    "ridge_clip_sc_k2":   "LR - SC (k=2)",
    "ridge_clip_sc_k5":   "LR - SC (k=5)",
    "ridge_clip_sc_k10":  "LR - SC (k=10)",
    "ridge_clip_sc_k20":  "LR - SC (k=20)",
    "ridge_clip_sc_k100": "LR - SC (k=100)",
    "ssc_ridge_k2":       "LR - SSC (k=2)",
    "ssc_ridge_k5":       "LR - SSC (k=5)",
    "ssc_ridge_k10":      "LR - SSC (k=10)",
    "ssc_ridge_k20":      "LR - SSC (k=20)",
}


def promote_semantic(df: pd.DataFrame, by=("dataset",)) -> pd.DataFrame:
    """Apply the paper's agreement rule to a results table.

    On the QA datasets, BASC and LR-SC use semantic agreement: within each
    group of `by`, every `*_sem_k*` method replaces its exact-match twin and
    takes its name (e.g. basc_sem_k2 -> basc_k2). On the math datasets (or if
    no semantic rows exist) the `*_sem_k*` rows are dropped.
    """
    parts = []
    for _, grp in df.groupby(list(by), sort=False):
        dataset = grp["dataset"].iloc[0] if "dataset" in grp else grp["test_ds"].iloc[0]
        is_sem = grp["method"].str.contains("_sem_k")
        if dataset in SEMANTIC_QA and is_sem.any():
            twins = set(grp.loc[is_sem, "method"].str.replace("_sem_k", "_k", regex=False))
            grp = grp[~grp["method"].isin(twins)].copy()
            grp["method"] = grp["method"].str.replace("_sem_k", "_k", regex=False)
        else:
            grp = grp[~is_sem].copy()
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def _parse_json_col(s):
    return np.asarray([np.nan if v is None else float(v) for v in json.loads(s)], dtype=float)


def pooled_reliability(df: pd.DataFrame, method: str, model_short=None, dataset=None):
    """Pool the per-row reliability bins of `method` (optionally restricted to
    one model / dataset). Returns (counts, mean confidence, accuracy) per bin,
    or None if there are no rows."""
    sub = df[df["method"] == method]
    if model_short is not None:
        sub = sub[sub["model_short"] == model_short]
    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]
    sub = sub[sub["rel_counts"].notna()]
    if sub.empty:
        return None
    counts = np.stack([np.asarray(json.loads(r), dtype=float) for r in sub["rel_counts"]])
    pred = np.stack([_parse_json_col(r) for r in sub["rel_pred"]])
    corr = np.stack([_parse_json_col(r) for r in sub["rel_correct"]])
    tot = counts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = np.where(tot > 0, np.nansum(counts * pred, axis=0) / tot, np.nan)
        mc = np.where(tot > 0, np.nansum(counts * corr, axis=0) / tot, np.nan)
    return tot, mp, mc
