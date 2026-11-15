"""Stage 1d: generate question-answer pairs from the domain passages (Balaguer et al. style pipeline).

- Test questions come only from TEST documents, train questions only from TRAIN documents.
- A pair is kept only if its answer appears word-for-word in the source passage (verifiable).
- Use a generator model that is NOT one of the three models being tested, otherwise the
  test set is biased towards that model. Default: Qwen2.5-7B-Instruct.

Output: data/processed/domain_train.jsonl, data/processed/domain_test.jsonl
Run:    python scripts/04_generate_domain_qa.py --train-target 1000
Afterwards, open domain_test.jsonl and check a random sample by hand.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from tqdm import tqdm

from kis.common import PROCESSED, load_config, read_jsonl, set_seed, write_jsonl
from kis.metrics import normalize
from kis.models import generate, load_model

PROMPT = """You are building an exam about financial and regulatory documents.
Read the passage and write ONE factual question that can be answered from this passage alone.
Rules:
- The answer must be a short span (1-8 words) copied EXACTLY from the passage.
- The question must make sense without seeing the passage (name the company or regulation).
- Do not ask about page numbers, headings or the table of contents.
Return only JSON: {{"question": "...", "answer": "..."}}

Passage:
{passage}"""


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if obj.get("question") and obj.get("answer") else None
    except json.JSONDecodeError:
        return None


def valid(qa: dict, passage: str) -> bool:
    a, q = normalize(qa["answer"]), normalize(qa["question"])
    return (0 < len(a.split()) <= 8 and a in normalize(passage) and a not in q)


def build(split: str, chunks: list[dict], target: int, model, tok, device, rng) -> list[dict]:
    pool = [c for c in chunks if c["split"] == split]
    rng.shuffle(pool)
    out = []
    for c in tqdm(pool, desc=split):
        if len(out) >= target:
            break
        msg = [{"role": "user", "content": PROMPT.format(passage=c["text"])}]
        qa = parse(generate(model, tok, device, msg, max_new_tokens=120).text)
        if qa and valid(qa, c["text"]):
            out.append({"id": f"domain-{split}-{len(out):04d}", "dataset": "domain", "split": split,
                        "type": "open", "question": qa["question"].strip(), "answers": [qa["answer"].strip()],
                        "source_chunk": c["chunk_id"], "source_doc": c["doc"], "evidence": c["text"]})
    if len(out) < target:
        print(f"[warn] only {len(out)} {split} items (target {target}); add more documents.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-target", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    chunks = read_jsonl(PROCESSED / "domain_chunks.jsonl")
    model, tok, device = load_model(args.generator, load_in_4bit=cfg["load_in_4bit"])

    test = build("test", chunks, cfg["domain"]["target_test_items"], model, tok, device, rng)
    train = build("train", chunks, args.train_target, model, tok, device, rng)
    write_jsonl(test, PROCESSED / "domain_test.jsonl")
    write_jsonl(train, PROCESSED / "domain_train.jsonl")
    print(f"domain: train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
