"""Quick checks that need no GPU or downloads. Run: python -m pytest -q"""
import numpy as np
import pandas as pd

from kis.metrics import exact_match, extract_choice, normalize, score_item
from kis.prompts import build_user_message
from kis.stats import binary_tests, continuous_tests
from kis.topsis import PROFILES, topsis


def test_normalize_keeps_numbers():
    assert normalize("The fee is 2.5%.") == "fee is 2.5%"
    assert exact_match("$1.2 billion", ["$1.2 Billion"]) == 1
    assert exact_match("Paris, France", ["Paris"]) == 0


def test_extract_choice():
    assert extract_choice("B", 4) == "B"
    assert extract_choice("(C) because...", 4) == "C"
    assert extract_choice("The answer is D.", 4) == "D"
    assert extract_choice("I am not sure", 4) is None


def test_truthfulqa_hallucination_flag():
    item = {"type": "mc", "dataset": "truthfulqa", "choices": ["x", "y"], "answer_letter": "A"}
    assert score_item(item, "B")["hallucinated"] == 1
    assert score_item(item, "A")["hallucinated"] == 0


def test_prompt_contains_choices():
    msg = build_user_message({"type": "mc", "question": "Q?", "choices": ["one", "two"]})
    assert "A. one" in msg and "B. two" in msg


def _fake(n=300, seed=0):
    rng = np.random.default_rng(seed)
    p = {"base": .60, "qlora": .70, "rag": .76, "cag": .75}
    lat = {"base": 300, "qlora": 310, "rag": 520, "cag": 350}
    rows = []
    for i in range(n):
        for s in p:
            rows.append({"item_id": i, "strategy": s, "correct": int(rng.random() < p[s]),
                         "latency_ms": rng.lognormal(np.log(lat[s]), .3)})
    return pd.DataFrame(rows)


def test_binary_and_continuous_tests():
    df = _fake()
    b = binary_tests(df, "correct")
    assert b["omnibus"]["p"] < 0.05 and len(b["pairwise"]) == 6
    c = continuous_tests(df, "latency_ms")
    assert c["omnibus"]["p"] < 0.05


def test_topsis_prefers_dominant_option():
    m = pd.DataFrame({"accuracy": [.6, .9], "hallucination_rate": [.4, .1], "latency_p50_ms": [500, 300],
                      "peak_gpu_gb": [10, 8], "cost_per_1k_usd": [1.0, .5]}, index=["worse", "better"])
    assert topsis(m, PROFILES["compliance"]).index[0] == "better"
