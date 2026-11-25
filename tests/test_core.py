"""Checks that need no GPU and no downloads. Run: python -m pytest -q

The pipeline tests use a tiny random model (tests/tiny.py), so they check the mechanics
(prompts, search, KV-cache reuse, result files), not answer quality.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from kis.metrics import exact_match, normalize
from kis.prompts import build_user_message, docs_prefix, question_suffix, target_text
from kis.retrieval import Retriever, chunk_text
from kis.stats import binary_tests, continuous_tests
from kis.tatqa import extract_final_answer, gold_text, parse_number, score_tatqa, serialize_context
from kis.topsis import PROFILES, topsis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


# ------------------------------------------------------------------ marking

def test_normalize_keeps_numbers():
    assert normalize("The fee is 2.5%.") == "fee is 2.5%"
    assert exact_match("$1.2 billion", ["$1.2 Billion"]) == 1


def test_parse_number():
    assert parse_number("$1,234.5") == (1234.5, "")
    assert parse_number("(3.1)") == (-3.1, "")
    assert parse_number("-0.2 %") == (-0.2, "percent")
    assert parse_number("2.5 million") == (2.5, "million")
    assert parse_number("$1,234 (in thousands)") == (1234.0, "thousand")
    assert parse_number("no number here") is None


def test_final_answer_is_taken_after_the_answer_label():
    assert extract_final_answer("Calculation: (16.6/93.8)*100\nAnswer: 17.7%") == "17.7%"
    assert extract_final_answer("17.7%") == "17.7%"


@pytest.mark.parametrize("item, prediction, strict, lenient", [
    ({"answer": 17.7, "scale": "percent"}, "Calculation: (16.6/93.8)*100\nAnswer: 17.7%", 1, 1),
    ({"answer": 17.7, "scale": "percent"}, "Answer: 0.177", 1, 1),          # same value, written as a fraction
    ({"answer": 17.7, "scale": "percent"}, "Answer: 17.7", 0, 1),           # right value, % missing
    ({"answer": 17.7, "scale": "percent"}, "Answer: 18.9%", 0, 0),
    ({"answer": -0.2, "scale": "percent"}, "Answer: (0.2)%", 1, 1),         # accounting negative
    ({"answer": -0.2, "scale": "percent"}, "Answer: 0.2%", 0, 0),           # wrong sign
    ({"answer": 1234, "scale": "thousand"}, "Answer: 1.234 million", 1, 1),  # same amount, other scale
    ({"answer": 1234, "scale": "thousand"}, "Answer: 1,234", 0, 1),         # scale missing
    ({"answer": 0, "scale": "thousand"}, "Answer: 8.85%", 0, 0),            # a percentage is not an amount
    ({"answer": ["1,568.6", "690.5"], "scale": ""}, "Answer: 690.5 ; 1,568.6", 1, 1),
    ({"answer": ["1,568.6", "690.5"], "scale": ""}, "Answer: 1,568.6", 0, 0),
    ({"answer": ["the modified retrospective method"], "scale": ""}, "Answer: The modified retrospective method.", 1, 1),
    ({"answer": ["the modified retrospective method"], "scale": ""}, "Answer: It used the modified retrospective method", 0, 1),
    ({"answer": ["2019"], "scale": ""}, "Answer: 2018", 0, 0),
    ({"answer": ["$41.4 million"], "scale": "million"}, "Answer: $41.4 million", 1, 1),
])
def test_tatqa_marking(item, prediction, strict, lenient):
    s = score_tatqa(item, prediction)
    assert (s["correct"], s["contains"]) == (strict, lenient)
    assert s["hallucinated"] == 1 - lenient


@pytest.mark.parametrize("prediction", ["Answer: I don't know", "I cannot determine this.", "Answer: Not provided", ""])
def test_refusals_are_not_hallucinations(prediction):
    s = score_tatqa({"answer": 5, "scale": ""}, prediction)
    assert s["abstained"] == 1 and s["hallucinated"] == 0 and s["correct"] == 0


def test_gold_text_and_target():
    item = {"answer": 17.7, "scale": "percent", "derivation": "(16.6/93.8) * 100", "question": "q"}
    item["gold_text"] = gold_text(item)
    assert item["gold_text"] == "17.7%"
    assert target_text(item) == "Calculation: (16.6/93.8) * 100\nAnswer: 17.7%"
    assert score_tatqa(item, target_text(item))["correct"] == 1
    assert gold_text({"answer": ["1,568.6"], "scale": "million"}) == "1,568.6 million"


# ------------------------------------------------------------------ prompts and search

def _contexts(n=6, block_size=3):
    names = ["Alpha", "Bravo", "Cedar", "Delta", "Ember", "Falcon", "Garnet", "Harbor"]
    out = []
    for i in range(n):
        table = [["Item", "2019", "2018"], [f"{names[i]} revenue", str(100 + i * 7), str(90 + i * 5)],
                 [f"{names[i]} costs", str(40 + i), str(35 + i)]]
        paras = [{"order": 1, "text": f"{names[i]} Corp adopted the {names[i].lower()} accounting method in 2019."}]
        out.append({"context_id": f"c{i}", "split": "test", "block": i // block_size,
                    "text": serialize_context(table, paras), "name": names[i]})
    return out


def _items(contexts):
    items = []
    for c in contexts:
        i = int(c["context_id"][1:])
        for j, (q, a, typ) in enumerate([(f"What was {c['name']} revenue in 2019?", [str(100 + i * 7)], "span"),
                                         (f"Which method did {c['name']} Corp adopt?", [f"the {c['name'].lower()} accounting method"], "span")]):
            it = {"id": f"tatqa-{c['context_id']}-{j}", "dataset": "tatqa", "split": "test", "type": "open",
                  "question": q, "answer": a, "scale": "", "answer_type": typ, "answer_from": "table" if j == 0 else "text",
                  "derivation": "", "context_id": c["context_id"], "block": c["block"]}
            it["gold_text"] = gold_text(it)
            items.append(it)
    return items


def test_prompt_is_documents_then_question():
    item = {"question": "What was revenue?"}
    assert build_user_message(item) == question_suffix(item)
    msg = build_user_message(item, "[Document 1]\nTable:\na | b")
    assert msg == docs_prefix("[Document 1]\nTable:\na | b") + question_suffix(item)
    assert msg.startswith("Use the following documents") and msg.endswith("Question: What was revenue?")


def test_chunks_keep_the_table_header():
    ctx = _contexts(1)[0]
    chunks = chunk_text(ctx["text"])
    assert chunks[0].startswith("Item | 2019 | 2018") and any("accounting method" in c for c in chunks)


def test_retriever_finds_the_right_extract():
    contexts = _contexts()
    r = Retriever(contexts, "tfidf")
    for it in _items(contexts):
        ids, ms = r.retrieve(it["question"], it["block"], k=1)
        assert ids == [it["context_id"]] and ms >= 0
    assert set(r.rank("revenue", 0)) == {"c0", "c1", "c2"}       # search stays inside the block


# ------------------------------------------------------------------ model mechanics (tiny model)

@pytest.fixture(scope="module")
def tiny():
    from tiny import tiny_model_and_tokenizer
    return tiny_model_and_tokenizer()


def test_cached_documents_give_identical_predictions(tiny):
    """CAG must be exact: reusing the KV cache gives the same next-token scores as reading everything again."""
    import torch
    from kis.models import PrefixCache, encode_prompt
    model, tok = tiny
    docs = docs_prefix("\n\n".join(c["text"] for c in _contexts()))
    cache, first = None, None
    for question in ["What was Alpha revenue in 2019?", "Which method did Delta Corp adopt?"]:
        prefix, suffix = encode_prompt(tok, docs, question_suffix({"question": question}))
        cache = cache or PrefixCache(model, "cpu", prefix, chunk_tokens=100)
        with torch.no_grad():
            full = model(input_ids=torch.tensor([prefix + suffix])).logits[0, -1]
            cached = model(input_ids=torch.tensor([suffix]), past_key_values=cache.cache).logits[0, -1]
        assert torch.allclose(full, cached, atol=1e-4)
        assert cache.reset() and cache.cache.get_seq_length() == len(prefix)
        if first is None:
            first = full
        else:
            assert not torch.allclose(first, full, atol=1e-4)     # a different question gives a different result


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_base_rag_cag_and_resume(tiny, tmp_path, monkeypatch):
    from kis.common import write_jsonl
    contexts = _contexts()
    items = _items(contexts)
    write_jsonl(contexts, tmp_path / "tatqa_contexts.jsonl")
    write_jsonl(items, tmp_path / "tatqa_test.jsonl")
    run = _load_script("03_run_eval")
    monkeypatch.setattr(run, "PROCESSED", tmp_path)
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    model, tok = tiny

    def call(*args):
        monkeypatch.setattr(sys, "argv", ["run", "--model", "llama8b", "--retriever", "tfidf", "--top-k", "1", *args])
        run.main((model, tok, "cpu"))

    call("--strategy", "base", "rag", "cag", "--limit", "8")
    rows = {s: list(csv.DictReader(open(tmp_path / "results" / "raw" / f"llama8b__{s}.csv"))) for s in ("base", "rag", "cag")}
    assert all(len(r) == 8 for r in rows.values())
    assert all(r["retrieval_hit"] == "1" for r in rows["rag"])                 # search found the right extract
    assert all(r["retrieval_hit"] == "" for r in rows["base"] + rows["cag"])
    assert [r["item_id"] for r in rows["base"]] == [r["item_id"] for r in rows["cag"]]   # same questions, same order
    assert int(rows["cag"][0]["prompt_tokens"]) > int(rows["rag"][0]["prompt_tokens"]) > int(rows["base"][0]["prompt_tokens"])

    call("--strategy", "base", "rag", "cag")                                   # resume: only the missing questions run
    rows = list(csv.DictReader(open(tmp_path / "results" / "raw" / "llama8b__cag.csv")))
    assert len(rows) == len(items) and len({r["item_id"] for r in rows}) == len(items)

    # --redo-truncated: an answer marked as cut off is generated again, the others are kept
    path = tmp_path / "results" / "raw" / "llama8b__base.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    cut = rows[2]["item_id"]
    rows[2]["output_tokens"] = "9999"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    call("--strategy", "base", "--redo-truncated")
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    assert len(rows) == len(items) and len({r["item_id"] for r in rows}) == len(items)
    assert next(r for r in rows if r["item_id"] == cut)["output_tokens"] != "9999"


def test_qlora_examples_train_only_on_the_reply(tiny):
    from kis.qlora import add_lora, build_examples, prepare_for_training, train
    model, tok = tiny
    contexts = _contexts()
    items = _items(contexts)
    examples, stats = build_examples(tok, items, contexts, n_docs=3, max_len=4000, seed=1)
    assert stats["kept"] == len(items) and stats["skipped_too_long"] == 0
    ex = examples[0]
    n_prompt = sum(label == -100 for label in ex["labels"])
    assert tok.decode(ex["input_ids"][n_prompt:]) == "Answer: " + items[0]["gold_text"] + "</s>"
    assert contexts[0]["text"][:40] in tok.decode(ex["input_ids"][:n_prompt])  # the gold extract is in the prompt
    from tiny import tiny_model_and_tokenizer
    fresh, _ = tiny_model_and_tokenizer(seed=1)
    log = train(add_lora(prepare_for_training(fresh), 4, 8, 0.0, ["q_proj", "v_proj"]), examples, epochs=4, lr=1e-2,
                grad_accum=2, log_every=1000)
    assert log["loss_end"] < log["loss_start"]


def test_reply_loss_matches_the_full_loss(tiny):
    """Scoring only the reply positions (to save GPU memory) gives the same loss as the full calculation."""
    from kis.qlora import _reply_loss, build_examples
    model, tok = tiny
    contexts = _contexts()
    examples, _ = build_examples(tok, _items(contexts), contexts, n_docs=3, max_len=4000, seed=1)
    ex = examples[0]
    ids, labels = torch.tensor([ex["input_ids"]]), torch.tensor([ex["labels"]])
    n_target = int((labels != -100).sum())
    with torch.no_grad():
        full = model(input_ids=ids, labels=labels).loss
        reply = _reply_loss(model, ids, n_target)
    assert torch.allclose(full, reply, atol=1e-5)


# ------------------------------------------------------------------ statistics and TOPSIS

def _fake(n=300, seed=0):
    rng = np.random.default_rng(seed)
    p = {"base": .05, "qlora": .62, "rag": .52, "cag": .40}
    lat = {"base": 1500, "qlora": 3900, "rag": 3800, "cag": 3000}
    return pd.DataFrame([{"item_id": i, "strategy": s, "correct": int(rng.random() < p[s]),
                          "latency_ms": rng.lognormal(np.log(lat[s]), .3)} for i in range(n) for s in p])


def test_binary_and_continuous_tests():
    df = _fake()
    b = binary_tests(df, "correct")
    assert b["omnibus"]["p"] < 0.05 and len(b["pairwise"]) == 6
    assert continuous_tests(df, "latency_ms")["omnibus"]["p"] < 0.05


def test_topsis_prefers_dominant_option():
    m = pd.DataFrame({"accuracy": [.4, .6], "f1": [.5, .7], "hallucination_rate": [.4, .1], "latency_p50_ms": [500, 300],
                      "peak_gpu_gb": [10, 8], "cost_per_1k_usd": [1.0, .5]}, index=["worse", "better"])
    assert topsis(m, PROFILES["compliance"]).index[0] == "better"


def test_strategy_report_runs_on_partial_results(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import IPython.display
    import kis.report as report
    rng = np.random.default_rng(0)
    (tmp_path / "raw").mkdir()
    for s, acc in [("base", .05), ("rag", .5)]:
        n = 504 if s == "base" else 300                       # RAG not finished yet
        pd.DataFrame({"item_id": range(n), "model": "llama8b", "strategy": s, "tag": "",
                      "answer_type": rng.choice(["span", "arithmetic"], n), "answer_from": "table",
                      "correct": (rng.random(n) < acc).astype(int), "contains": 0, "f1": 0.0,
                      "hallucinated": 0, "abstained": 0, "retrieval_hit": 1 if s == "rag" else None,
                      "latency_ms": 1000.0, "prompt_tokens": 100, "output_tokens": 20}).to_csv(tmp_path / "raw" / f"llama8b__{s}.csv", index=False)
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)
    assert report.show("rag").loc["Llama-3.1-8B", "Questions"] == 300
    assert report.show("cag") is None                          # nothing to show yet, no error
    assert (tmp_path / "figures" / "rag_overview.png").exists()
