"""Step 1: download TAT-QA and build the train / dev / test files and the knowledge base.

TAT-QA's official splits are kept, so no report extract appears in more than one split:
  train -> QLoRA training      dev -> retrieval check and choosing settings      test -> final results

A "context" is one report extract (a table + its paragraphs) with about 6 questions.
Test and dev contexts are put into groups ("blocks") of `block_size`. Each block is one
small knowledge base: RAG searches it, CAG preloads it.

Output (data/processed/):
  tatqa_contexts.jsonl   every report extract as text, with its split and block
  tatqa_train.jsonl  tatqa_dev.jsonl  tatqa_test.jsonl   the questions

Run: python scripts/01_prepare_tatqa.py
"""
from __future__ import annotations

import json
import random
import time
import urllib.request
from collections import Counter

from kis.common import DATA, PROCESSED, load_config, write_jsonl
from kis.tatqa import gold_text, serialize_context


def download(url: str, path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, path)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  download failed ({e}); retrying...")
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"Could not download {url}")


def build(split: str, raw: list[dict], n_contexts: int, block_size: int | None, rng: random.Random):
    raw = list(raw)
    rng.shuffle(raw)                       # fixed seed -> the same sample every time
    contexts, items = [], []
    for i, c in enumerate(raw[:n_contexts]):
        cid = c["table"]["uid"]
        block = i // block_size if block_size else -1
        contexts.append({"context_id": cid, "split": split, "block": block,
                         "text": serialize_context(c["table"]["table"], c["paragraphs"])})
        for q in c["questions"]:
            item = {"id": f"tatqa-{q['uid']}", "dataset": "tatqa", "split": split, "type": "open",
                    "question": q["question"].strip(), "answer": q["answer"], "scale": q.get("scale") or "",
                    "answer_type": q["answer_type"], "answer_from": q["answer_from"],
                    "derivation": (q.get("derivation") or "").strip(), "context_id": cid, "block": block}
            item["gold_text"] = gold_text(item)
            items.append(item)
    return contexts, items


def main():
    cfg = load_config()
    t = cfg["tatqa"]
    raw_dir = DATA / "raw" / "tatqa"
    raw = {}
    for split, fname in t["files"].items():
        path = raw_dir / fname
        download(f"{t['base_url']}/{fname}", path)
        raw[split] = json.loads(path.read_text(encoding="utf-8"))
        print(f"{split}: {len(raw[split])} report extracts, "
              f"{sum(len(c['questions']) for c in raw[split])} questions available")

    rng = random.Random(cfg["seed"])
    plan = [("test", t["test_contexts"], t["block_size"]),
            ("dev", t["dev_contexts"], t["block_size"]),
            ("train", t["train_contexts"], None)]
    all_contexts = []
    print("\nselected:")
    for split, n, block_size in plan:
        contexts, items = build(split, raw[split], n, block_size, rng)
        all_contexts += contexts
        write_jsonl(items, PROCESSED / f"tatqa_{split}.jsonl")
        types = dict(Counter(i["answer_type"] for i in items))
        print(f"  {split}: {len(contexts)} extracts, {len(items)} questions, answer types {types}")
        if block_size:
            for b in sorted({c["block"] for c in contexts}):
                chars = sum(len(c["text"]) for c in contexts if c["block"] == b)
                print(f"     block {b}: {sum(c['block'] == b for c in contexts)} extracts, "
                      f"about {chars // 3:,} tokens")
    write_jsonl(all_contexts, PROCESSED / "tatqa_contexts.jsonl")

    ids = {s: {c["context_id"] for c in all_contexts if c["split"] == s} for s in ("train", "dev", "test")}
    assert not (ids["train"] & ids["test"]) and not (ids["train"] & ids["dev"]) and not (ids["dev"] & ids["test"])
    print("\ncheck passed: train, dev and test share no report extracts")


if __name__ == "__main__":
    main()
