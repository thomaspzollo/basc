#!/usr/bin/env bash
# Step 5 (GPU): baselines that need additional model calls.
#   - verbalized confidence, prompt stem 6, for every (model, dataset)
#   - all 10 stems on the prompt-selection grid (4 models x GSM8K, TriviaQA;
#     DeepSeek-R1-Distill-Llama-8B only on GSM8K)
#   - P(Yes) scores of 32 pool responses per question for LR-SSC
#   - base-model scores for the five Qwen3 models (appendix)
# Token probabilities (TP, ATP) come from step 2; TT-SC and the supervised
# reference are computed in step 6.
source "$(dirname "$0")/config.sh"

for model in "${MODELS[@]}"; do
  for ds in $(datasets_for "$model"); do
    $PYTHON scripts/baselines/verbalized_confidence.py --model "$model" --dataset "$ds" --stem_idx 6 --n_device "$N_GPU"
  done
done

VC_GRID_MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-8B" "nvidia/Nemotron-Cascade-8B-Thinking" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
for model in "${VC_GRID_MODELS[@]}"; do
  for ds in gsm8k trivia_qa; do
    [[ " $(datasets_for "$model") " == *" $ds "* ]] || continue
    for stem in 0 1 2 3 4 5 7 8 9; do
      $PYTHON scripts/baselines/verbalized_confidence.py --model "$model" --dataset "$ds" --stem_idx "$stem" --n_device "$N_GPU"
    done
  done
done

for model in "${MODELS[@]}"; do
  for ds in $(datasets_for "$model"); do
    $PYTHON scripts/baselines/ssc_scores.py --model "$model" --dataset "$ds" --k_ssc 32 --n_device "$N_GPU"
  done
done

for model in "${QA_MODELS[@]:0:5}"; do        # the five Qwen3 models
  for ds in "${MATH_DATASETS[@]}" "${QA_DATASETS[@]}"; do
    $PYTHON scripts/baselines/base_model_scores.py --model "$model" --dataset "$ds"
  done
done
