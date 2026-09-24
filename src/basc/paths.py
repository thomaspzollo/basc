"""Canonical locations and file names for every artifact the pipeline writes.

All per-(model, dataset) artifacts live in

    {OUTPUTS_DIR}/{model_short}/{dataset}/

where `model_short` is the Hugging Face model name after the last "/". File
names encode the sampling configuration: a probe *pool* of `k` responses per
question at temperature `temp` has stem `n_{n}_temp_{temp}_k_{k}`, and the
single deployed (anchor) response at temperature 0.6 is stored under the
canonical pool's stem with suffix `_out_df_temp_0.6_k_1`.

The three top-level directories default to `outputs/`, `results/` and
`figures/` in the repository root and can be overridden with the environment
variables BASC_OUTPUTS_DIR, BASC_RESULTS_DIR and BASC_FIGURES_DIR.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = Path(os.environ.get("BASC_OUTPUTS_DIR", REPO_ROOT / "outputs"))
RESULTS_DIR = Path(os.environ.get("BASC_RESULTS_DIR", REPO_ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("BASC_FIGURES_DIR", REPO_ROOT / "figures"))

# Canonical sampling configuration used throughout the paper.
N = 1000            # questions per dataset
POOL_TEMP = 0.7     # probe pool temperature
POOL_K = 100        # probe pool size per question
TEST_TEMP = 0.6     # deployed (anchor) response temperature
TEST_K = 1          # one deployed response per question

# Embedding backbones: "own" = the reasoning model itself, "gemma" = the
# external model used for the black-box variant (BASC-L-bb).
EMBED_VARIANT_INFIX = {"own": "", "gemma": "_gemma"}
BLACKBOX_EMBED_MODEL = "google/gemma-3-4b-it"


def model_short(model: str) -> str:
    return model.split("/")[-1]


def run_dir(model: str, dataset: str, outputs_dir: str | Path | None = None) -> Path:
    root = Path(outputs_dir) if outputs_dir is not None else OUTPUTS_DIR
    return root / model_short(model) / dataset


def pool_stem(n: int, temp: float, k: int) -> str:
    return f"n_{n}_temp_{temp}_k_{k}"


def _variant_infix(variant: str) -> str:
    if variant not in EMBED_VARIANT_INFIX:
        raise ValueError(f"variant={variant!r} not in {sorted(EMBED_VARIANT_INFIX)}")
    return EMBED_VARIANT_INFIX[variant]


# --- Raw generations (scripts/pipeline/generate.py) -------------------------

def raw_generations(model, dataset, *, n, temp, k, outputs_dir=None) -> Path:
    """Pickled list of dicts, one per sampled response (includes token logprobs)."""
    return run_dir(model, dataset, outputs_dir) / f"{pool_stem(n, temp, k)}.pkl"


# --- Probe pool (scripts/pipeline/process_generations.py) -------------------

def pool_answers(model, dataset, *, n, temp, k, outputs_dir=None) -> Path:
    """(id, answer) for every pool response, ordered by (id, sample_idx)."""
    return run_dir(model, dataset, outputs_dir) / f"{pool_stem(n, temp, k)}_SLIM_train_df.pkl"


def pool_responses(model, dataset, *, n, temp, k, outputs_dir=None) -> Path:
    """(id, sample_idx, answer, response) for every pool response."""
    return run_dir(model, dataset, outputs_dir) / f"{pool_stem(n, temp, k)}_ROLLOUTS_train_df.pkl"


def pool_embeddings(model, dataset, *, n, temp, k, variant="own", outputs_dir=None) -> Path:
    """{ids, sample_idx, embeddings} for every pool response (scripts/pipeline/embed.py)."""
    infix = _variant_infix(variant)
    return run_dir(model, dataset, outputs_dir) / f"{pool_stem(n, temp, k)}_ROLLOUTS_qr{infix}_embeddings.npz"


# --- Deployed responses ------------------------------------------------------

def test_stem(*, n, pool_temp, pool_k, test_temp=TEST_TEMP, test_k=TEST_K) -> str:
    return f"{pool_stem(n, pool_temp, pool_k)}_out_df_temp_{test_temp}_k_{test_k}"


def test_slim(model, dataset, *, n, pool_temp, pool_k, test_temp=TEST_TEMP, test_k=TEST_K,
              outputs_dir=None) -> Path:
    name = f"{pool_stem(n, pool_temp, pool_k)}_SLIM_test_df_temp_{test_temp}_k_{test_k}.pkl"
    return run_dir(model, dataset, outputs_dir) / name


def test_responses(model, dataset, *, n, pool_temp, pool_k, test_temp=TEST_TEMP, test_k=TEST_K,
                   outputs_dir=None) -> Path:
    """Per-question CSV: question, ground truth, deployed response/answer, token
    log-probability features, correctness, and empirical self-consistency."""
    stem = test_stem(n=n, pool_temp=pool_temp, pool_k=pool_k, test_temp=test_temp, test_k=test_k)
    return run_dir(model, dataset, outputs_dir) / f"{stem}.csv"


def test_embeddings(model, dataset, *, n, pool_temp, pool_k, variant="own",
                    test_temp=TEST_TEMP, test_k=TEST_K, outputs_dir=None) -> Path:
    """{ids, embeddings} for every deployed (question, response) pair."""
    infix = _variant_infix(variant)
    stem = test_stem(n=n, pool_temp=pool_temp, pool_k=pool_k, test_temp=test_temp, test_k=test_k)
    return run_dir(model, dataset, outputs_dir) / f"{stem}_question_response{infix}_embeddings.npz"


def verbal_confidence(model, dataset, *, stem_idx, n=N, pool_temp=POOL_TEMP, pool_k=POOL_K,
                      outputs_dir=None) -> Path:
    stem = test_stem(n=n, pool_temp=pool_temp, pool_k=pool_k)
    return run_dir(model, dataset, outputs_dir) / f"{stem}_verbal_conf_stem_{stem_idx}.csv"


# --- Agreement / baseline caches --------------------------------------------

def anchor_semantic_agreement(model, dataset, *, n, temp, k, test_temp=TEST_TEMP,
                              outputs_dir=None) -> Path:
    """(N, K) bidirectional-entailment agreement between each deployed answer
    and every pool response (QA datasets only)."""
    name = f"{pool_stem(n, temp, k)}_ANCHOR_T{test_temp}_semantic_agree.npz"
    return run_dir(model, dataset, outputs_dir) / name


def ssc_scores(model, dataset, *, n, temp, k, k_ssc, outputs_dir=None) -> Path:
    """Per-pool-response P(Yes) scores and SSC targets (Huang et al., 2025)."""
    return run_dir(model, dataset, outputs_dir) / f"{pool_stem(n, temp, k)}_ROLLOUTS_ssc_k{k_ssc}.csv"


def base_model_scores(model, dataset, outputs_dir=None) -> Path:
    """Deployed responses scored under the paired pre-trained (Base) model."""
    stem = model_short(model).lower().replace("-", "_")
    return run_dir(model, dataset, outputs_dir) / f"{stem}_{dataset}_with_base_targets.csv"
