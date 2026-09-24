#!/usr/bin/env bash
# Step 1 (GPU): sample responses with vLLM.
#   - probe pool:        k=100 responses per question at T=0.7
#   - deployed response: k=1 response per question at T=0.6
#   - temperature ablation: k=20 pools at T=0.5 and T=0.9 (two models)
source "$(dirname "$0")/config.sh"

for model in "${MODELS[@]}"; do
  for ds in $(datasets_for "$model"); do
    $PYTHON scripts/pipeline/generate.py --model "$model" --dataset "$ds" --k 100 --temperature 0.7 --n_device "$N_GPU"
    $PYTHON scripts/pipeline/generate.py --model "$model" --dataset "$ds" --k 1   --temperature 0.6 --n_device "$N_GPU"
  done
done

for model in "${TEMP_ABLATION_MODELS[@]}"; do
  for t in "${TEMP_ABLATION_TEMPS[@]}"; do
    for ds in "${MATH_DATASETS[@]}" "${QA_DATASETS[@]}"; do
      $PYTHON scripts/pipeline/generate.py --model "$model" --dataset "$ds" --k 20 --temperature "$t" --n_device "$N_GPU"
    done
  done
done
