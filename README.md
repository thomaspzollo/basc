# BASC: Binary Agreement for Self-consistency Calibration

Code for the paper **Binary Agreement for Efficient Unsupervised Confidence Calibration of Reasoning LLMs**.

BASC produces a calibrated confidence for a reasoning LLM's answer from a **single generation**,
without correctness labels. On an unlabeled calibration set it samples one extra *probe*
response per question and records whether the probe's answer agrees with the answer being
calibrated (the *anchor*). These binary agreements supervise two stages:

1. **Stage 1: score.** A binomial-logistic regressor from the anchor's embedding to its agreement rate
   (**BASC-L**), or any existing score such as token probability (**BASC-TP**).
2. **Stage 2: recalibration.** Binary isotonic regression of agreement on the frozen score, fit on
   held-out questions. This stage carries a distribution-free calibration guarantee with respect to
   latent self-consistency.

At deployment only the one response is generated; no probes are needed.

---

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Using BASC on your own data](#using-basc-on-your-own-data)
- [Reproducing the paper](#reproducing-the-paper)
- [Where each figure and table comes from](#where-each-figure-and-table-comes-from)
- [Methods and baselines](#methods-and-baselines)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)
- [License](#license)

## Repository layout

```
src/basc/                    the library (pip install -e .)
  stage1.py                  Stage 1: binomial-logistic score on binary agreements
  stage2.py                  Stage 2: binary isotonic recalibration (nearest-knot extension)
  sampling.py                20/20/60 calibration/test split, anchor + probe draws
  experiment.py              one trial of every method on a shared split
  baselines.py               RR-SC, RR-SSC, supervised two-stage reference
  answers.py                 \boxed{} extraction, correctness, exact-match agreement
  semantic.py                semantic agreement (bidirectional NLI entailment) for QA
  verbal_confidence.py       verbalized-confidence prompts and parsing
  metrics.py                 ECE1/ECE2, Brier, AUROC, reliability bins, selective prediction
  data.py                    BASCData: loads the cached artifacts of one (model, dataset)
  datasets.py, generation.py HF dataset loaders; vLLM generation/embedding and the QA prompt
  paths.py, roster.py, shift.py   file layout, models/datasets, distribution-shift splits
scripts/
  01_generate.sh ... 07_figures.sh   end-to-end drivers (run in order)
  config.sh                  model/dataset roster used by the drivers
  pipeline/                  generate.py, process_generations.py, embed.py, semantic_agreement.py
  baselines/                 verbalized_confidence.py, ssc_scores.py, base_model_scores.py
  experiments/               run_main.py, run_basc_tp.py, run_one_stage.py, run_stage1_scaling.py,
                             run_recalibrators.py, run_shift.py, run_selection.py
notebooks/                   one notebook per paper figure/table group (read results/, write figures/)
```

The repository contains code only. Generated data goes to `outputs/`, experiment results to
`results/`, and figures and LaTeX tables to `figures/`. All three are git-ignored and can be moved
with the environment variables `BASC_OUTPUTS_DIR`, `BASC_RESULTS_DIR` and `BASC_FIGURES_DIR`.

## Installation

Python ≥ 3.10. From the repository root:

```bash
# Everything (generation needs CUDA GPUs):
pip install -e ".[generation,notebooks]"

# Only the CPU experiments and notebooks, given existing outputs/:
pip install -e ".[notebooks]"

# Exact versions used for the paper (Python 3.11, CUDA 12.8):
pip install -r requirements.txt && pip install -e .
```

Models and datasets are downloaded from the Hugging Face Hub. Set `HF_HOME` to a disk with a few
hundred GB free before running the pipeline. `pip install -e ".[dev]" && pytest tests/` runs a
few fast sanity checks that need no model outputs.

## Using BASC on your own data

BASC needs, for each calibration question, an embedding of the (question, anchor response) pair
and binary agreements between the anchor answer and one or more probe answers:

```python
import numpy as np
from basc.stage1 import fit_predict_binomial_logistic
from basc.stage2 import fit_binary_isotonic

# Disjoint calibration splits, all unlabeled:
#   X_s1 (n1, D): anchor embeddings;  B_s1 (n1, k-1): agreement of each anchor with k-1 probes
#   X_s2 (n2, D): anchor embeddings;  B_s2 (n2,):     agreement with a single probe
# X_test (m, D): embeddings of deployed (question, response) pairs

m = np.full(len(X_s1), B_s1.shape[1])   # probes per question
c = B_s1.sum(axis=1)                    # agreements per question
U = fit_predict_binomial_logistic(X_s1, m, c, np.concatenate([X_s2, X_test]))   # Stage 1
stage2 = fit_binary_isotonic(U[:len(X_s2)], B_s2)                               # Stage 2
confidence = stage2.predict(U[len(X_s2):])
```

For **BASC-TP**, skip Stage 1 and use any existing score, e.g.
`fit_binary_isotonic(token_prob_cal, B_cal).predict(token_prob_test)`.

Agreement is `basc.answers.probe_agree_matrix` (exact match; used for math) or the
bidirectional-entailment rule in `basc.semantic` (used for free-form QA).

## Reproducing the paper

The paper evaluates 9 reasoning models on GSM8K and PolyMath, and 6 of them also on SciQ,
TriviaQA and WebQuestions: 36 model/dataset combinations, 1000 questions each (`src/basc/roster.py`).
Run the drivers from the repository root, in order. Each driver loops over the full roster, and
each step can also be run for a single (model, dataset) by calling the Python script it wraps.

| Step | Driver | Hardware | What it does | Writes to `outputs/{model}/{dataset}/` |
|---|---|---|---|---|
| 1 | `scripts/01_generate.sh` | GPU | Samples a probe pool (k=100, T=0.7) and one deployed response (T=0.6) per question with vLLM; also the k=20 pools at T=0.5/0.9 for the temperature ablation | `n_1000_temp_0.7_k_100.pkl`, `n_1000_temp_0.6_k_1.pkl` |
| 2 | `scripts/02_process.sh` | CPU | Slims the pickles; computes token-probability features, correctness and empirical self-consistency | `*_SLIM_train_df.pkl`, `*_ROLLOUTS_train_df.pkl`, `*_out_df_temp_0.6_k_1.csv` |
| 3 | `scripts/03_embed.sh` | GPU | Embeds every (question, response) with the reasoning model itself and with Gemma-3-4B-it (black-box variant) | `*_question_response[_gemma]_embeddings.npz`, `*_ROLLOUTS_qr[_gemma]_embeddings.npz` |
| 4 | `scripts/04_semantic.sh` | GPU | Semantic agreement between each deployed answer and its pool (QA only; DeBERTa-v3 NLI) | `*_ANCHOR_T0.6_semantic_agree.npz` |
| 5 | `scripts/05_baselines.sh` | GPU | Verbalized confidence (10 prompts), P(Yes) scores for RR-SSC, base-model scores | `*_verbal_conf_stem_{i}.csv`, `*_ROLLOUTS_ssc_k32.csv`, `*_with_base_targets.csv` |
| 6 | `scripts/06_experiments.sh` | CPU | Every experiment: splits, fits and metrics for all methods, 10 seeds | `results/*.csv` |
| 7 | `scripts/07_figures.sh` | CPU | Runs the notebooks | `figures/*.png`, `figures/tables/*.tex` |

Steps 1–5 use vLLM with tensor parallelism (`N_GPU`, default 4). `PYTHON` and `N_GPU` can be set
in the environment or in `scripts/config.sh`. Steps 6–7 need only CPUs; step 6 takes about 40
minutes on a 256-core machine and step 7 about two minutes.

**Disk.** The raw generation pickles take about 285 GB (2–16 GB per probe pool). Only step 2 reads
them, so they can be deleted once it has finished; all other artifacts take about 75 GB.

## Where each figure and table comes from

| Paper | Notebook | Results file(s) | Produced by |
|---|---|---|---|
| Fig. 1 (method illustration) | `01_method_figure` | `main.csv` | `run_main.py` |
| Table 1 (main results) | `02_main_results` | `main.csv`, `main_blackbox.csv`, `basc_tp.csv` | `run_main.py` (`--variant gemma` for BASC-L-bb), `run_basc_tp.py --setting in_distribution` |
| Fig. 2 (a) reliability, (b) accuracy vs. ECE$_1$ (`final_5_figure_a_b.png`) | `03_analysis_figure` | `main.csv` | `run_main.py` |
| Fig. 3 (a) training decodes (`final_5_figure_c_d_e.png`) | `03_analysis_figure` | `main.csv`, `one_stage.csv`, `basc_tp.csv` | `run_main.py`, `run_one_stage.py`, `run_basc_tp.py` |
| Fig. 3 (b) Stage-1 estimator | `03_analysis_figure` | `stage1_scaling.csv` | `run_stage1_scaling.py` |
| Fig. 3 (c) Stage-2 recalibrator (isotonic vs. temperature scaling, Platt, histogram binning) | `03_analysis_figure` | `stage2_recalibrators.csv` | `run_recalibrators.py` |
| Table 2 (distribution shift) | `04_distribution_shift` | `shift_{language,math,qa}.csv`, `basc_tp_shift.csv` | `run_shift.py`, `run_basc_tp.py --setting shift` |
| Fig. 4 (selective prediction) | `05_selective_prediction` | `selection.csv` | `run_selection.py` |
| Table 3 (VC prompt selection) | `06_appendix_vc_prompts` | `outputs/**/*_verbal_conf_stem_*.csv` | `baselines/verbalized_confidence.py` |
| Tables 4–5 (base models) | `07_appendix_base_models` | `outputs/**/*_with_base_targets.csv`, `main.csv` | `baselines/base_model_scores.py` |
| Table 6 (accuracy) | `02_main_results` | `main.csv` | `run_main.py` |
| Appendix: per-model / per-dataset results and reliability diagrams | `02_main_results` | `main.csv`, `basc_tp.csv` | `run_main.py`, `run_basc_tp.py` |
| Appendix: temperature ablation | `08_appendix_ablations` | `temp_ablation_T*.csv` | `run_main.py --pool_temp ... --pool_k ...` |
| Appendix: exact vs. semantic agreement | `08_appendix_ablations` | `main.csv` | `run_main.py` |
| Appendix: distribution shift averaged over all shifts (bootstrap CIs) and per pair | `04_distribution_shift` | as Table 2 | as Table 2 |
| Appendix: selection by dataset / model family | `05_selective_prediction` | `selection.csv` | `run_selection.py` |

The exact flags for every results file are in `scripts/06_experiments.sh`. The calibration error shown in
Tables 1–2, Figures 2–3 and the shift figures is set by `MAIN_METRIC` in
`notebooks/paper_utils.py` (`"ECE1"`, as in the paper, or `"ECE2"`; the environment variable `BASC_MAIN_METRIC`
overrides it).

## Methods and baselines

Method keys used in the results files, with the paper's names:

| Key | Paper | Implementation |
|---|---|---|
| `basc_k{k}` | BASC-L (k decodes per Stage-1 question) | `stage1.fit_predict_binomial_logistic` + `stage2.fit_binary_isotonic` |
| `basc_sem_k{k}` | BASC-L with semantic agreement; replaces `basc_k{k}` on QA in every table | same, with `semantic.load_anchor_semantic` |
| `tp_iso_dcal` (`run_basc_tp.py`) | BASC-TP | `stage2.fit_binary_isotonic` on token probability |
| `basc_k{k}` in `main_blackbox.csv` | BASC-L-bb (Gemma-3-4B-it embeddings) | `run_main.py --variant gemma` |
| `logprob` | TP: geometric-mean token probability of the response | `scripts/pipeline/process_generations.py` |
| `ans_logprob` | ATP: same, for the `\boxed{}` answer tokens | `scripts/pipeline/process_generations.py` |
| `verbal_conf_stem6` | VC: verbalized confidence, prompt 6 | `scripts/baselines/verbalized_confidence.py` |
| `ridge_clip_sc[_sem]_k{k}` | RR-SC: ridge on empirical self-consistency | `baselines.fit_predict_ridge_clip` |
| `ssc_ridge_k{k}` | RR-SSC: ridge on soft self-consistency (Huang et al., 2025) | `baselines.compute_ssc_matrix`, `scripts/baselines/ssc_scores.py` |
| `onestage_binomial_k{k}` | 1-stage (BASC-L Stage 1 on all calibration data) | `scripts/experiments/run_one_stage.py` |
| `oracle_sc_k100` | TT-SC: test-time self-consistency, 100 samples | `experiment.oracle_sc` |
| `sup_logreg_iso_k1` | Sup. (2-stage, correctness labels) | `baselines.fit_predict_supervised_two_stage` |

Every calibration-set method in a trial uses the same split and the same anchors and probes
(`experiment.run_id_trial` / `run_shift_trial`). Metrics use 12 equal-mass bins for ECE and are
averaged over 10 random 40/60 calibration/test splits (seeds 0–9).

## Reproducibility notes

- **BLAS threads.** The Stage-1 logistic regressions are fit with lbfgs on high-dimensional
  embeddings, and their solutions depend slightly on the OpenBLAS thread count. In our checks,
  changing it moved per-seed ECE by up to about 0.01. The paper's results were computed with 16 threads, and every
  script in `scripts/experiments/` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
  `MKL_NUM_THREADS` to 16 unless they are already set. With the pinned `requirements.txt` and the
  paper's `outputs/`, step 6 reproduces the paper's results files exactly.
- **Generation.** vLLM sampling is seeded (seed 12345) but not bit-for-bit deterministic across
  hardware and batch composition, so regenerated responses and embeddings will differ slightly
  from the paper's.
- **PolyMath language blocks.** The language-shift splits assume question ids 0–124 are English,
  125–249 Spanish, and so on (`basc.datasets._load_polymath` with n=1000).

## Citation

```bibtex
@article{basc2026,
  title = {Binary Agreement for Efficient Unsupervised Confidence Calibration of Reasoning LLMs},
  year  = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
