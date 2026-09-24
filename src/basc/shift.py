"""Distribution-shift settings (Section 4.3).

Language shift: PolyMath's 8 languages occupy consecutive blocks of 125
question ids (see basc.datasets). Each split calibrates on four languages and
tests on the other four; eight splits in total.
Domain shift: calibrate on one dataset, test on another of the same type.
"""
from __future__ import annotations

import numpy as np

LANG_NAMES = ["en", "es", "fr", "pt", "it", "ru", "ar", "id"]
LANG_BLOCK = 125

# split id -> (training languages, test languages), as indices into LANG_NAMES
LANGUAGE_SPLITS = {
    1: ([0, 1, 2, 3], [4, 5, 6, 7]),
    2: ([4, 5, 6, 7], [0, 1, 2, 3]),
    3: ([1, 2, 3, 4], [0, 5, 6, 7]),
    4: ([0, 5, 6, 7], [1, 2, 3, 4]),
    5: ([0, 1, 2, 5], [3, 4, 6, 7]),
    6: ([3, 4, 6, 7], [0, 1, 2, 5]),
    7: ([0, 1, 5, 6], [2, 3, 4, 7]),
    8: ([2, 3, 4, 7], [0, 1, 5, 6]),
}

MATH_PAIRS = [("polymath", "gsm8k"), ("gsm8k", "polymath")]           # (train, test)
QA_PAIRS = [
    ("webq", "trivia_qa"),
    ("sciq", "trivia_qa"),
    ("trivia_qa", "webq"),
    ("sciq", "webq"),
    ("webq", "sciq"),
    ("trivia_qa", "sciq"),
]


def lang_ids(lang_idx: list[int]) -> np.ndarray:
    """PolyMath question ids for the given language indices."""
    return np.concatenate([np.arange(k * LANG_BLOCK, (k + 1) * LANG_BLOCK) for k in lang_idx])


def lang_label(lang_idx: list[int]) -> str:
    return "_".join(LANG_NAMES[k] for k in lang_idx)
