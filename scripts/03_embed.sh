#!/usr/bin/env bash
# Step 3 (GPU): embed (question, response) pairs.
#   own    the reasoning model's own embedding (BASC-L)
#   gemma  Gemma-3-4B-it (black-box BASC-L-bb)
source "$(dirname "$0")/config.sh"

for model in "${MODELS[@]}"; do
  for ds in $(datasets_for "$model"); do
    for variant in own gemma; do
      $PYTHON scripts/pipeline/embed.py --model "$model" --dataset "$ds" --target test --variant "$variant" --n_device "$N_GPU"
      $PYTHON scripts/pipeline/embed.py --model "$model" --dataset "$ds" --target pool --variant "$variant" --n_device "$N_GPU"
    done
  done
done

for model in "${TEMP_ABLATION_MODELS[@]}"; do
  for t in "${TEMP_ABLATION_TEMPS[@]}"; do
    for ds in "${MATH_DATASETS[@]}" "${QA_DATASETS[@]}"; do
      $PYTHON scripts/pipeline/embed.py --model "$model" --dataset "$ds" --target pool \
        --pool_temp "$t" --pool_k 20 --n_device "$N_GPU"
    done
  done
done
