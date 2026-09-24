"""Models and datasets evaluated in the paper (36 model/dataset combinations).

All nine reasoning models are evaluated on the two math datasets. The free-form
QA datasets are evaluated only for the six models that reliably follow the
\\boxed{} answer format on them.
"""
from __future__ import annotations

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "nvidia/OpenReasoning-Nemotron-7B",
    "nvidia/Nemotron-Cascade-8B-Thinking",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

QA_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "nvidia/Nemotron-Cascade-8B-Thinking",
]

QWEN_MODELS = [m for m in MODELS if m.startswith("Qwen/")]

MATH_DATASETS = ["gsm8k", "polymath"]
QA_DATASETS = ["sciq", "trivia_qa", "webq"]
DATASETS = MATH_DATASETS + QA_DATASETS

# Temperature ablation (Appendix): extra probe pools of k=20 responses.
TEMP_ABLATION_MODELS = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B-Thinking-2507"]
TEMP_ABLATION_POOLS = [(0.5, 20), (0.7, 100), (0.9, 20)]   # (temperature, k)

# Paired pre-trained models for the base-model appendix.
BASE_MODELS = {
    "Qwen/Qwen3-0.6B": "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B": "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Thinking-2507": "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B": "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-14B": "Qwen/Qwen3-14B-Base",
}


def datasets_for(model: str) -> list[str]:
    """Datasets evaluated for `model` in the paper."""
    return DATASETS if model in QA_MODELS else MATH_DATASETS


def combos() -> list[tuple[str, str]]:
    """All 36 (model, dataset) pairs, dataset-major as in the paper sweeps."""
    return [(m, d) for d in DATASETS for m in MODELS if d in datasets_for(m)]
