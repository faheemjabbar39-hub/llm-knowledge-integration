"""TAT-QA helpers: turning a report extract into text, and marking model answers.

Marking follows the logic of the official TAT-QA metric (a number is right only if
value x scale matches the gold answer), adapted to free-text model output.
"""
from __future__ import annotations

import re
from collections import Counter

from kis.metrics import normalize

SCALE = {"": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9, "percent": 0.01}
_UNIT_WORDS = {"%": "percent", "percent": "percent", "thousand": "thousand", "thousands": "thousand",
               "million": "million", "millions": "million", "billion": "billion", "billions": "billion",
               "k": "thousand", "m": "million", "mn": "million", "bn": "billion", "b": "billion"}

# ---------------------------------------------------------------- context -> text


def serialize_context(table: list[list[str]], paragraphs: list[dict]) -> str:
    """One report extract as plain text: the table (one row per line), then its paragraphs."""
    rows = [" | ".join(str(c).strip() for c in row) for row in table]
    paras = [p["text"].strip() for p in sorted(paragraphs, key=lambda p: p["order"])]
    return "Table:\n" + "\n".join(rows) + "\n\nText:\n" + "\n".join(paras)


def format_docs(texts: list[str]) -> str:
    return "\n\n".join(f"[Document {i}]\n{t}" for i, t in enumerate(texts, 1))


# ---------------------------------------------------------------- numbers

_NUM = re.compile(
    r"(?P<lead>[-−–]\s*)?[$€£]?\s*(?P<open>\()?\s*(?P<inner>[-−–]\s*)?[$€£]?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?|\.\d+)\s*(?P<close>\))?\s*"
    r"(?P<unit>%|percent\b|thousands?\b|millions?\b|billions?\b|bn\b|mn\b|[kmb]\b)?",
    re.IGNORECASE,
)
_IN_UNITS = re.compile(r"\bin\s+(thousands?|millions?|billions?)\b", re.IGNORECASE)


def parse_number(text: str) -> tuple[float, str] | None:
    """First number in the text as (value, scale). Handles $1,234.5, (3.1) = -3.1, 17.7%, 2.5 million."""
    m = _NUM.search(str(text))
    if not m:
        return None
    value = float(m.group("num").replace(",", ""))
    if m.group("lead") or m.group("inner") or (m.group("open") and m.group("close")):
        value = -value
    unit = (m.group("unit") or "").lower()
    scale = _UNIT_WORDS.get(unit, "")
    if not scale:
        later = _IN_UNITS.search(str(text))
        if later:
            scale = _UNIT_WORDS[later.group(1).lower()]
    return value, scale


def _is_numeric_span(span: str) -> bool:
    """True for '1,568.6', '$0.5 million', '(3.1)', '12%'; False for 'fiscal 2019' or free text."""
    s = str(span).strip()
    words = re.sub(r"[$€£(),%]", "", s).split()
    if not words:
        return False
    try:
        float(words[0].replace("−", "-"))
    except ValueError:
        return False
    return len(words) == 1 or (len(words) == 2 and words[1].lower() in _UNIT_WORDS)


def _fmt_number(v: float) -> str:
    return f"{v:,.4f}".rstrip("0").rstrip(".")


def _with_scale(text: str, scale: str) -> str:
    if not scale:
        return text
    if scale == "percent":
        return text if "%" in text else text + "%"
    return text if scale in text.lower() else f"{text} {scale}"


# ---------------------------------------------------------------- gold answers


def gold_spans(item: dict) -> list[str]:
    a = item["answer"]
    return [str(x) for x in a] if isinstance(a, list) else [str(a)]


def gold_text(item: dict) -> str:
    """The reference answer as one readable string (used in results files and as the QLoRA target)."""
    a, scale = item["answer"], item.get("scale") or ""
    if isinstance(a, list):
        return " ; ".join(_with_scale(s, scale) if _is_numeric_span(s) else s for s in map(str, a))
    try:
        return _with_scale(_fmt_number(float(a)), scale)
    except (TypeError, ValueError):
        return _with_scale(str(a), scale)


# ---------------------------------------------------------------- model answers

_ABSTAIN = re.compile(
    r"i\s+(?:do\s+not|don['’]?t|cannot|can['’]?t|am\s+unable\s+to)\s+(?:know|answer|determine|find|tell)",
    re.IGNORECASE,
)
# Looser wording only counts as "I don't know" when the whole answer is short.
_ABSTAIN_SHORT = re.compile(
    r"\bnot\s+(?:provided|available|mentioned|specified|stated|given|enough|possible)\b"
    r"|\b(?:no|insufficient)\s+(?:information|data|context)\b|\bunknown\b|^n/?a$",
    re.IGNORECASE,
)


def extract_final_answer(text: str) -> str:
    """The text after the last 'Answer:' (first line of it); if there is none, the last non-empty line."""
    text = str(text or "").strip()
    parts = re.split(r"answer\s*:", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        tail = parts[-1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def is_abstention(answer: str) -> bool:
    a = answer.strip()
    if a == "" or _ABSTAIN.search(a):
        return True
    return len(a.split()) <= 8 and bool(_ABSTAIN_SHORT.search(a))


def _split_candidates(answer: str, n_gold: int) -> list[list[str]]:
    """Ways to cut a list answer into items: '; ', ', ', and ', ... and' (an item may itself contain 'and')."""
    if n_gold == 1:
        return [[answer.strip()]]
    cands = []
    for pattern in (r";", r",\s+(?!\d{3}\b)", r",\s+(?:and\s+)?(?!\d{3}\b)|\s+and\s+(?=[^,]*$)",
                    r"\s+and\s+|,\s+(?!\d{3}\b)"):
        parts = [p.strip(" .") for p in re.split(pattern, answer) if p.strip(" .")]
        if len(parts) > 1 and parts not in cands:
            cands.append(parts)
    return cands or [[answer.strip()]]


_YEAR = re.compile(r"(?:19[89]\d|20[0-3]\d)")


def _numbers(text: str) -> list[str]:
    out = []
    for m in _NUM.finditer(str(text)):
        n = m.group(0).strip()
        if n.count("(") != n.count(")"):           # a bracket from the sentence, e.g. "(for 2018)"
            n = n.strip("()")                       # balanced "(47,969)" stays negative
        out.append(n.strip())
    return out


def _match_one(pred: str, gold: str, scale: str) -> tuple[bool, bool]:
    """(strict, lenient) for one predicted part against one gold span."""
    if _is_numeric_span(gold):
        g, p = parse_number(gold), parse_number(pred)
        if g is None or p is None:
            return False, False
        g_val, g_inline = g
        p_val, p_scale = p
        g_unit = SCALE[g_inline] if g_inline else SCALE[scale]  # absolute size of one gold unit
        p_in_gold_units = p_val * SCALE[p_scale] / g_unit
        g_scale = g_inline or scale
        same_kind = (p_scale == "percent") == (g_scale == "percent") or not p_scale
        # same value to 2 decimals, scale included; a percentage never matches a money amount
        strict = same_kind and abs(p_in_gold_units - g_val) <= 0.005 + 1e-9
        if g_val == 0:
            strict = strict and abs(p_val) <= 0.005
        if float(g_val).is_integer() and "." not in gold:      # whole numbers (years, counts) must match exactly
            tol = 1e-6
        else:                                                   # decimals: allow rounding differences (0.5%)
            tol = max(0.005, 0.005 * abs(g_val))
        wrong_kind = p_scale == "percent" and g_scale != "percent"
        lenient = strict or (not wrong_kind and (abs(p_val - g_val) <= tol or abs(p_in_gold_units - g_val) <= tol))
        return strict, lenient
    ng, np_ = normalize(gold), normalize(pred)
    return (ng == np_ and ng != ""), (ng != "" and ng in np_)


def _token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def score_tatqa(item: dict, prediction: str) -> dict:
    """Per-question scores written to the results CSV.

    correct      strict: every gold answer matched, with the right scale (thousand / million / %)
    contains     lenient: right value even if the scale or extra words differ
    f1           token overlap for text answers; equals `correct` for numeric answers
    abstained    the model said it does not know
    hallucinated the model gave an answer and it was wrong (not abstained and not lenient-correct)
    """
    answer = extract_final_answer(prediction)
    golds, scale = gold_spans(item), item.get("scale") or ""
    if is_abstention(answer):
        return {"final_answer": answer, "correct": 0, "contains": 0, "f1": 0.0, "abstained": 1, "hallucinated": 0}

    strict_all = False
    for parts in _split_candidates(answer, len(golds)):   # strict: one predicted item per gold answer
        if len(parts) != len(golds):
            continue
        remaining, ok = list(parts), True
        for g in golds:
            hit = next((p for p in remaining if _match_one(p, g, scale)[0]), None)
            if hit is None:
                ok = False
                break
            remaining.remove(hit)
        if ok:
            strict_all = True
            break
    if len(golds) == 1 and _is_numeric_span(golds[0]) and ";" in answer:
        strict_all = False                            # one value was asked for, several were given

    # lenient: right values even if the scale or extra words differ, but no extra or alternative values
    if all(_is_numeric_span(g) for g in golds) and (len(golds) > 1 or ";" in answer):
        nums = _numbers(answer)
        if not any(_YEAR.fullmatch(g.strip()) for g in golds):   # "(for 2018)" labels are not extra values
            nums = [n for n in nums if not _YEAR.fullmatch(n)]
        lenient_all = bool(nums) and all(any(_match_one(n, g, scale)[1] for n in nums) for g in golds) \
            and all(any(_match_one(n, g, scale)[1] for g in golds) for n in nums)
    else:
        lenient_all = all(any(_match_one(p, g, scale)[1] for p in [answer] + [x for c in
                          _split_candidates(answer, len(golds)) for x in c]) for g in golds)
    lenient_all = lenient_all or strict_all

    numeric = all(_is_numeric_span(g) for g in golds)
    f1 = float(strict_all) if numeric else max(_token_f1(answer, " ".join(golds)), float(strict_all))
    return {"final_answer": answer, "correct": int(strict_all), "contains": int(lenient_all),
            "f1": round(f1, 4), "abstained": 0, "hallucinated": int(not lenient_all)}
