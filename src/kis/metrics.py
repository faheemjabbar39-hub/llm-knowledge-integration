"""Scoring. Same functions are used for every condition (analysis layer)."""
from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace (SQuAD-style).
    Keeps digits, %, $, and decimal points because numbers matter in finance."""
    text = text.lower()
    keep = set("%$.")
    text = "".join(ch for ch in text if ch not in string.punctuation or ch in keep)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)  # remove full stops that aren't decimals
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: list[str]) -> int:
    p = normalize(prediction)
    return int(any(p == normalize(g) for g in golds))


def contains_match(prediction: str, golds: list[str]) -> int:
    """Lenient check: gold appears inside the answer. Reported alongside EM, never instead of it."""
    p = normalize(prediction)
    return int(any(normalize(g) and normalize(g) in p for g in golds))


_LETTER = re.compile(r"\b([A-J])\b")


def extract_choice(prediction: str, n_choices: int) -> str | None:
    """Pull the chosen option letter out of a model answer like 'B', '(B)', 'Answer: B'."""
    valid = string.ascii_uppercase[:n_choices]
    s = prediction.strip()
    if s[:1].upper() in valid and (len(s) == 1 or not s[1].isalpha()):
        return s[0].upper()
    for m in _LETTER.finditer(s.upper()):
        if m.group(1) in valid:
            return m.group(1)
    return None


def score_item(item: dict, prediction: str) -> dict:
    """Returns the per-item scores written to the results CSV."""
    if item["type"] == "mc":
        choice = extract_choice(prediction, len(item["choices"]))
        correct = int(choice == item["answer_letter"])
        out = {"pred_choice": choice, "correct": correct, "contains": correct}
        # TruthfulQA: picking a wrong option = repeating a misconception = hallucination
        out["hallucinated"] = (1 - correct) if item["dataset"] == "truthfulqa" else None
        return out
    return {
        "pred_choice": None,
        "correct": exact_match(prediction, item["answers"]),
        "contains": contains_match(prediction, item["answers"]),
        "hallucinated": None,
    }
