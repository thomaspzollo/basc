"""Verbalized-confidence baseline: prompts and output parsing.

Ten candidate prompt stems (0-4 written by GPT-5.2, 5-9 by Gemini-3) were
compared in the appendix; stem 6 performed best and is used in all main
experiments (VERBAL_CONF_STEM_IDX).
"""
from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd

VERBAL_CONF_STEM_IDX = 6

_BOXED_PROB_RE = re.compile(r"""\\boxed\s*\{\s*([01](?:\.\d+)?)\s*\}""", flags=re.IGNORECASE | re.DOTALL)
_BOX_PROB_RE = re.compile(r"""\\box\s*\{\s*([01](?:\.\d+)?)\s*\}""", flags=re.IGNORECASE | re.DOTALL)
_FLOAT_RE = re.compile(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")


def parse_prob(text: str) -> Optional[float]:
    """Probability from \\boxed{p} (or \\box{p}, a bare number, or the first
    number in [0, 1] in the text); None if nothing parses."""
    if text is None:
        return None
    t = text.strip()
    for regex in (_BOXED_PROB_RE, _BOX_PROB_RE):
        m = regex.search(t)
        if m:
            try:
                v = float(m.group(1))
                return v if 0.0 <= v <= 1.0 else None
            except Exception:
                pass
    try:
        v = float(t)
        if 0.0 <= v <= 1.0:
            return v
    except Exception:
        pass
    m = _FLOAT_RE.search(t)
    if not m:
        return None
    try:
        v = float(m.group(0))
        return v if 0.0 <= v <= 1.0 else None
    except Exception:
        return None


VERBAL_CONF_STEMS = [
    (
        "You are a calibration assistant.\n"
        "Task: estimate the probability that the given ANSWER is correct for the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Write at most THREE short sentences of reasoning (keep it brief).\n"
        "2) On a NEW LINE, output ONLY the probability in the form \\boxed{p} where p is a decimal in [0,1].\n"
        "   - Examples: \\boxed{0.03}, \\boxed{0.62},  \\boxed{0.81}, \\boxed{0.97}\n"
        "   - Do not output those example scores exactly, try to be precise about the exact probability of this answer being correct\n"
        "3) Do NOT write anything after the \\boxed{...} line.\n"
    ),
    (
        "You are a calibration assistant.\n"
        "Your job is to judge how likely the provided ANSWER is to be correct for the given QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Give no more than a few brief sentences explaining your estimate.\n"
        "2) Then, on a NEW LINE, output ONLY a probability as \\boxed{p}, where p is a decimal in [0,1].\n"
        "   - Examples of format only: \\boxed{0.14}, \\boxed{0.58}, \\boxed{0.89}\n"
        "   - Do not copy those example values; choose the most precise probability you can.\n"
        "3) Do NOT write anything after the \\boxed{...} line.\n"
    ),
    (
        "You are an uncertainty estimation assistant.\n"
        "Task: assess the probability that the ANSWER correctly resolves the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Provide a very brief explanation of your reasoning.\n"
        "2) On a NEW LINE, output ONLY the final probability in the exact form \\boxed{p}, with p in [0,1].\n"
        "   - Format examples: \\boxed{0.07}, \\boxed{0.44}, \\boxed{0.93}\n"
        "   - Do not reuse the example numbers; give your own precise estimate.\n"
        "3) Do NOT include any text after the \\boxed{...} line.\n"
    ),
    (
        "You are a confidence scoring assistant.\n"
        "Task: estimate the chance that the supplied ANSWER is correct given the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Write up to four or five concise sentences of reasoning.\n"
        "2) On a separate NEW LINE, output ONLY \\boxed{p}, where p is a decimal probability between 0 and 1.\n"
        "   - Example formatting: \\boxed{0.11}, \\boxed{0.69}, \\boxed{0.95}\n"
        "   - Do not use those exact values; report the most accurate probability you can.\n"
        "3) Do NOT add anything after the \\boxed{...} line.\n"
    ),
    (
        "You are a probability calibration assistant.\n"
        "Task: determine how likely it is that the ANSWER is correct for the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Use at most THREE short reasoning sentences.\n"
        "2) Then on a NEW LINE, output ONLY the probability as \\boxed{p}, where p is a decimal in [0,1].\n"
        "   - Formatting examples: \\boxed{0.05}, \\boxed{0.51}, \\boxed{0.98}\n"
        "   - Do not repeat those example values; instead provide a precise estimate.\n"
        "3) Do NOT write any additional text after the \\boxed{...} line.\n"
    ),
    (
        "You are a reliability assessment assistant.\n"
        "Task: evaluate the probability that the provided ANSWER is the correct response to the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Write no more than 3-6 brief sentences of reasoning.\n"
        "2) On a NEW LINE, output ONLY the probability in the form \\boxed{p}, where p is a decimal in [0,1].\n"
        "   - Examples: \\boxed{0.09}, \\boxed{0.37}, \\boxed{0.75}, \\boxed{0.91}\n"
        "   - Do not copy these example values; provide your own precise probability estimate.\n"
        "3) Do NOT include any additional text after the \\boxed{...} line.\n"
    ),
    (
        "You are a fact-checking and calibration model.\n"
        "Task: determine the likelihood that the ANSWER accurately solves the given QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Use at most FOUR concise sentences to explain your reasoning.\n"
        "2) On a NEW LINE, output ONLY the probability using the format \\boxed{p} (p is a decimal in [0,1]).\n"
        "   - Format examples: \\boxed{0.18}, \\boxed{0.42}, \\boxed{0.66}, \\boxed{0.99}\n"
        "   - Do not reuse these example scores; instead, be as accurate as possible for this specific case.\n"
        "3) Do NOT write any text after the \\boxed{...} line.\n"
    ),
    (
        "You are an answer verification assistant.\n"
        "Task: calculate the probability that the ANSWER is correct given the context of the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Provide at most THREE short sentences of logic or reasoning.\n"
        "2) Then, on a NEW LINE, output ONLY the final probability as \\boxed{p}, where p is in [0,1].\n"
        "   - Examples: \\boxed{0.02}, \\boxed{0.25}, \\boxed{0.55}, \\boxed{0.88}\n"
        "   - Do not use these exact numbers; choose the most precise decimal for the current answer.\n"
        "3) Do NOT add any comments or text after the \\boxed{...} line.\n"
    ),
    (
        "You are an expert calibration system.\n"
        "Task: assign a probability score representing how likely the ANSWER is to be the correct solution for the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Limit your reasoning to at most 4-6 brief sentences.\n"
        "2) On a separate NEW LINE, output ONLY \\boxed{p}, with p being a decimal between 0 and 1.\n"
        "   - Formatting examples: \\boxed{0.13}, \\boxed{0.49}, \\boxed{0.72}, \\boxed{0.94}\n"
        "   - Do not repeat these example values; provide your own calculated probability.\n"
        "3) Do NOT include any characters or text after the \\boxed{...} line.\n"
    ),
    (
        "You are a truth-estimation assistant.\n"
        "Task: predict the probability that the given ANSWER is a correct and valid response to the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Write up to FIVE short sentences explaining your thought process.\n"
        "2) On a NEW LINE, output ONLY the probability in the exact form \\boxed{p}, where p is a decimal in [0,1].\n"
        "   - Format examples: \\boxed{0.06}, \\boxed{0.31}, \\boxed{0.64}, \\boxed{0.92}\n"
        "   - Do not output the example values; give a precise estimate for the answer provided.\n"
        "3) Do NOT write anything at all after the \\boxed{...} line.\n"
    ),
]


def build_verbal_conf_prompt(tokenizer, model_name: str, question: str, answer: str,
                             stem_idx: int = VERBAL_CONF_STEM_IDX) -> str:
    """Ask the model for the probability that `answer` is correct. The
    reasoning trace of the original response is not shown."""
    stem = VERBAL_CONF_STEMS[stem_idx]
    user = f"{stem}\nQUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
    if "Qwen3" in model_name:
        messages = [{"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
    if "Llama" in model_name:   # DeepSeek-R1-Distill-Llama-8B: stem also as system message
        messages = [{"role": "system", "content": stem}, {"role": "user", "content": user}]
    else:
        messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_verbal_conf_scores(path, ids: pd.Series) -> Optional[np.ndarray]:
    """Parsed confidences from a verbal-confidence CSV, aligned to `ids`
    (NaN where the output could not be parsed); None if the file is missing."""
    if not os.path.exists(path):
        print(f"[verbal_conf] missing: {path}")
        return None
    vc_df = pd.read_csv(path)
    left = pd.DataFrame({"id": ids.astype(str).to_numpy()})
    right = vc_df[["id", "verbal_confidence"]].copy()
    right["id"] = right["id"].astype(str)
    merged = left.merge(right, on="id", how="left")
    return pd.to_numeric(merged["verbal_confidence"], errors="coerce").to_numpy(dtype=float)


def impute_missing_with_mean(vc: np.ndarray) -> np.ndarray:
    """Unparseable outputs get the mean of the parsed confidences."""
    mask = np.isfinite(vc)
    return np.where(mask, vc, float(vc[mask].mean()))
