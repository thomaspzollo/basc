#!/usr/bin/env bash
# Step 4 (GPU): semantic (bidirectional-entailment) agreement between each
# deployed answer and its probe pool, for the QA datasets.
source "$(dirname "$0")/config.sh"
DEVICE="${DEVICE:-cuda:0}"

for model in "${QA_MODELS[@]}"; do
  for ds in "${QA_DATASETS[@]}"; do
    $PYTHON scripts/pipeline/semantic_agreement.py --model "$model" --dataset "$ds" --device "$DEVICE" --fp16
  done
done

for model in "${TEMP_ABLATION_MODELS[@]}"; do
  for t in "${TEMP_ABLATION_TEMPS[@]}"; do
    for ds in "${QA_DATASETS[@]}"; do
      $PYTHON scripts/pipeline/semantic_agreement.py --model "$model" --dataset "$ds" \
        --pool_temp "$t" --pool_k 20 --test_pool_temp 0.7 --test_pool_k 100 --device "$DEVICE" --fp16
    done
  done
done
