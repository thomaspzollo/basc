#!/usr/bin/env bash
# Step 2 (CPU): slim the raw generations into per-question tables and compute
# token-probability features, correctness and empirical self-consistency.
source "$(dirname "$0")/config.sh"

for model in "${MODELS[@]}"; do
  for ds in $(datasets_for "$model"); do
    $PYTHON scripts/pipeline/process_generations.py --model "$model" --dataset "$ds"
  done
done

for model in "${TEMP_ABLATION_MODELS[@]}"; do
  for t in "${TEMP_ABLATION_TEMPS[@]}"; do
    for ds in "${MATH_DATASETS[@]}" "${QA_DATASETS[@]}"; do
      $PYTHON scripts/pipeline/process_generations.py --model "$model" --dataset "$ds" \
        --pool_temp "$t" --pool_k 20 --pool_only
    done
  done
done
