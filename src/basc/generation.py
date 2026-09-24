"""vLLM wrappers for response generation and embedding, and the QA prompt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .answers import extract_last_boxed

QA_PROMPT_STEM = "Please reason step by step, and put your final answer within \\boxed{}."

# Every model in the paper uses the same single-turn prompt with thinking enabled.
SUPPORTED_MODEL_FAMILIES = ("Qwen3", "deepseek", "Nemotron")


@dataclass(frozen=True)
class GenerateConfig:
    model: str
    n_device: int
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    logprobs: int = 1
    min_p: float = 0.0
    top_k: int = 20


def load_vllm_generate(cfg: GenerateConfig) -> tuple[Any, LLM, SamplingParams]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    llm = LLM(model=cfg.model, tensor_parallel_size=cfg.n_device)
    sampling = SamplingParams(
        n=1,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        logprobs=cfg.logprobs,
        seed=cfg.seed,
        min_p=cfg.min_p,
        top_k=cfg.top_k,
    )
    return tokenizer, llm, sampling


def load_vllm_embed(model: str, n_device: int, *, enforce_eager: bool = False) -> LLM:
    return LLM(model=model, task="embed", tensor_parallel_size=n_device, enforce_eager=enforce_eager)


def build_qa_prompt(tokenizer, model_name: str, question: str) -> str:
    if not any(f in model_name for f in SUPPORTED_MODEL_FAMILIES):
        raise ValueError(f"No prompt template for {model_name}; supported families: "
                         f"{SUPPORTED_MODEL_FAMILIES}")
    messages = [{"role": "user", "content": f"{QA_PROMPT_STEM}\nQuestion: {question}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def gather_prompts(tokenizer, items: list[dict], model_name: str) -> list[str]:
    return [build_qa_prompt(tokenizer, model_name, ex["question"]) for ex in items]


def generate_k(llm: LLM, prompts: list[str], sampling_base: SamplingParams, *, k: int) -> list:
    """One vLLM call that draws k responses per prompt."""
    params = SamplingParams(
        n=k,
        temperature=sampling_base.temperature,
        top_p=sampling_base.top_p,
        max_tokens=sampling_base.max_tokens,
        logprobs=sampling_base.logprobs,
        seed=sampling_base.seed,
        min_p=sampling_base.min_p,
        top_k=sampling_base.top_k,
    )
    return llm.generate(prompts, params)


def rows_from_generation(
    items: list[dict],
    prompts: list[str],
    results: list,
    *,
    answer_parser: Callable[[str], str] = extract_last_boxed,
) -> tuple[list[dict], float]:
    """One row per sampled response:
    id, question, ground_truth, prompt, sample_idx, response, answer, logprobs.
    Returns (rows, fraction of responses with no extractable answer)."""
    rows: list[dict] = []
    num_bad = 0
    for ex, prompt, res in zip(items, prompts, results):
        for j, out in enumerate(res.outputs):
            ans = answer_parser(out.text)
            if not ans:
                num_bad += 1
            rows.append({
                "id": ex["id"],
                "question": ex["question"],
                "ground_truth": ex["ground_truth"],
                "prompt": prompt,
                "sample_idx": j,
                "response": out.text,
                "answer": ans,
                "logprobs": out.logprobs,
            })
    return rows, (num_bad / len(rows)) if rows else 0.0


def embed_texts(llm: LLM, texts: list[str]) -> np.ndarray:
    """Embed every text in a single vLLM call (vLLM batches internally)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    outputs = llm.embed(texts)
    return np.vstack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in outputs]).astype(np.float32)
