"""
Utility functions for tokenizing schema names and entity texts for readability assessment.

The `tokenize` module built into Python is not used below, as it is designed for tokenizing Python source code. Instead, this module requires a simple separation of dataset-specific identifiers and free-text values, such as Snake_Case, CamelCase, Kebab-Case, and plain-text cell contents.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Any

_CAMEL_SPLIT = re.compile(r"(?<!^)(?=[A-Z])")

def split_identifier(identifier: str) -> List[str]:
    if not identifier:
        return []
    tmp = re.sub(r"[_\-]+", " ", str(identifier))
    parts: List[str] = []
    for p in tmp.split():
        parts.extend(_CAMEL_SPLIT.sub(" ", p).split())
    return [t.lower() for t in parts if t]

def split_text(text: Any) -> List[str]:
    if text is None:
        return []
    tokens = re.split(r"[^\w]+", str(text))
    return [t.lower() for t in tokens if t]

def detect_case_style(label: str) -> str:
    if not label:
        return "other"
    s = str(label)
    if s.islower():
        return "snake" if "_" in s else "lower"
    if s.isupper():
        return "upper"
    if _CAMEL_SPLIT.search(s):
        return "camel"
    if any(c.islower() for c in s) and any(c.isupper() for c in s):
        return "mixed"
    return "other"

def compute_case_consistency_scores(labels: List[str]) -> Dict[str, float]:
    if not labels:
        return {}
    styles = [detect_case_style(l) for l in labels]
    dominant = Counter(styles).most_common(1)[0][0]
    scores: Dict[str, float] = {}
    for label, style in zip(labels, styles):
        if style == dominant:
            scores[label] = 1.0
        elif {style, dominant}.issubset({"lower", "snake", "camel"}):
            scores[label] = 0.5
        else:
            scores[label] = 0.0
    return scores
