"""Load evaluation datasets from the Hugging Face Hub in a canonical format.

Every loader returns a list of dicts with keys
  id            str, position of the question in the loaded list
  question      str
  ground_truth  str (math, SciQ) or list[str] of accepted aliases (TriviaQA, WebQ)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import load_dataset

POLYMATH_LANGS = ["en", "es", "fr", "pt", "it", "ru", "ar", "id"]
POLYMATH_SPLIT = "low"


def load_dataset_items(
    name: str,
    *,
    n: int,
    polymath_langs: Optional[str] = None,
    polymath_split: str = POLYMATH_SPLIT,
) -> List[Dict[str, Any]]:
    name = name.strip()
    if name == "gsm8k":
        return _load_gsm8k(n=n)
    if name == "polymath":
        return _load_polymath(n=n, polymath_langs=polymath_langs, polymath_split=polymath_split)
    if name == "sciq":
        return _load_sciq(n=n)
    if name == "trivia_qa":
        return _load_trivia_qa(n=n)
    if name == "webq":
        return _load_webq(n=n)
    raise ValueError(f"Unknown dataset: {name}")


def _load_gsm8k(*, n: int) -> List[Dict[str, Any]]:
    ds = load_dataset("openai/gsm8k", "main")["train"]
    items = []
    for i, ex in enumerate(ds):
        q = ex.get("question")
        if not q:
            continue
        gt = ex.get("answer", "").split("####")[-1].strip()
        items.append({"id": str(i), "question": str(q), "ground_truth": gt})
        if len(items) >= n:
            break
    return items[:n]


def _load_polymath(*, n: int, polymath_langs: Optional[str], polymath_split: str) -> List[Dict[str, Any]]:
    """Questions are taken language by language (125 per language in the "low"
    split), so with n=1000 ids 0-124 are English, 125-249 Spanish, etc. The
    language-shift experiments rely on this block layout."""
    if polymath_langs:
        langs = [x.strip() for x in polymath_langs.split(",") if x.strip()]
    else:
        langs = POLYMATH_LANGS
    items = []
    for lang in langs:
        ds = load_dataset("Qwen/PolyMath", lang, split=polymath_split or POLYMATH_SPLIT)
        for ex in ds:
            q, ans = ex.get("question"), ex.get("answer")
            if not q or not ans:
                continue
            items.append({"id": str(len(items)), "question": str(q), "ground_truth": str(ans)})
            if len(items) >= n:
                return items[:n]
    return items[:n]


def _load_sciq(*, n: int) -> List[Dict[str, Any]]:
    """Free-form SciQ: the question alone, graded against the correct answer."""
    ds = load_dataset("sciq", split="test")
    items = []
    for i, ex in enumerate(ds):
        q, gt = ex.get("question"), ex.get("correct_answer")
        if not q or not gt:
            continue
        distractors = [ex.get("distractor1"), ex.get("distractor2"), ex.get("distractor3")]
        if len([d for d in distractors if isinstance(d, str) and d.strip()]) != 3:
            continue
        items.append({"id": str(i), "question": str(q), "ground_truth": str(gt)})
        if len(items) >= n:
            break
    return items[:n]


def _load_trivia_qa(*, n: int) -> List[Dict[str, Any]]:
    def _answers(ex):
        ans = ex.get("answer") or {}
        aliases = ans.get("aliases") or []
        value = ans.get("value")
        out = []
        if isinstance(aliases, list):
            out.extend([str(a) for a in aliases if a])
        if isinstance(value, str) and value:
            out.append(value)
        if not out and isinstance(ex.get("answers"), list):
            out.extend([str(a) for a in ex["answers"] if a])
        return list(dict.fromkeys(out))   # unique, order-preserving

    ds = load_dataset("trivia_qa", "rc.nocontext")["validation"]
    items = []
    for i, ex in enumerate(ds):
        q = ex.get("question")
        if not q:
            continue
        answers = _answers(ex)
        if not answers:
            continue
        items.append({"id": str(i), "question": str(q), "ground_truth": answers})
        if len(items) >= n:
            break
    return items[:n]


def _load_webq(*, n: int) -> List[Dict[str, Any]]:
    ds = load_dataset("stanfordnlp/web_questions", split="test")
    items = []
    for i, ex in enumerate(ds):
        q, answers = ex.get("question"), ex.get("answers")
        if not q or not answers:
            continue
        items.append({"id": str(i), "question": str(q), "ground_truth": answers})
        if len(items) >= n:
            break
    return items[:n]
