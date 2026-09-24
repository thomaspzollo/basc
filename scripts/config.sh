#!/usr/bin/env bash
# Shared configuration for the pipeline drivers (sourced, not executed).
# Mirrors src/basc/roster.py. Override any variable from the environment, e.g.
#   PYTHON=/path/to/python N_GPU=2 bash scripts/01_generate.sh

PYTHON="${PYTHON:-python}"
N_GPU="${N_GPU:-4}"            # tensor-parallel GPUs for vLLM

MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B-Thinking-2507"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "nvidia/OpenReasoning-Nemotron-7B"
  "nvidia/Nemotron-Cascade-8B-Thinking"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)
QA_MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B-Thinking-2507"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "nvidia/Nemotron-Cascade-8B-Thinking"
)
MATH_DATASETS=(gsm8k polymath)
QA_DATASETS=(sciq trivia_qa webq)

# Temperature ablation: extra k=20 probe pools for two models.
TEMP_ABLATION_MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B-Thinking-2507")
TEMP_ABLATION_TEMPS=(0.5 0.9)

# datasets_for MODEL -> the paper's datasets for that model
datasets_for() {
  local m
  for m in "${QA_MODELS[@]}"; do
    if [[ "$m" == "$1" ]]; then echo "${MATH_DATASETS[@]} ${QA_DATASETS[@]}"; return; fi
  done
  echo "${MATH_DATASETS[@]}"
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repository root
set -euo pipefail
