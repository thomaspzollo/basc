"""BASC: Binary Agreement for Self-consistency Calibration.

Unsupervised confidence calibration for reasoning LLMs from a single
generation. BASC learns a confidence score from binary anchor-vs-probe
agreement observations (Stage 1) and recalibrates it with binary isotonic
regression on held-out agreement observations (Stage 2).

Module map:
  paths           canonical file locations for every artifact
  roster          models / datasets used in the paper
  datasets        loading the evaluation datasets from the Hugging Face Hub
  generation      vLLM generation + embedding wrappers and the QA prompt
  answers         answer extraction, correctness and exact-match agreement
  semantic        NLI (bidirectional entailment) agreement for QA
  data            BASCData: all cached artifacts for one (model, dataset)
  sampling        calibration/test splits and anchor + probe draws
  stage1          binomial-logistic score model (Stage 1)
  stage2          binary isotonic recalibration (Stage 2)
  baselines       RR-SC / RR-SSC ridge baselines, supervised reference
  experiment      one trial of every method on a shared split
  shift           distribution-shift split definitions
  metrics         ECE, Brier, AUROC, reliability bins, selective prediction
  verbal_confidence  verbalized-confidence prompts and parsing
"""

__version__ = "1.0.0"
