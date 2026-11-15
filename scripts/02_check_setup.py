"""Step 2: two quick checks before any long run (no language model is loaded).

A. Search check. Many TAT-QA questions do not name the company ("What was revenue in
   2019?"), so this measures how often the correct report extract is among the top results
   inside its block. It uses the DEV questions, so the test set stays untouched.
B. Size check. Counts the tokens of every knowledge block with each model's own tokenizer,
   to confirm that CAG can preload the block inside the model's limit.

Output: tables on screen and results/setup_check.json
Run:    python scripts/02_check_setup.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from kis.common import PROCESSED, RESULTS, load_config, read_jsonl
from kis.prompts import docs_prefix, question_suffix
from kis.retrieval import Retriever
from kis.strategies import KnowledgeBase


def block_sizes(cfg: dict) -> dict:
    """Tokens per knowledge block for each model (tokenizer only, a small download)."""
    from transformers import AutoTokenizer

    from kis.models import encode_prompt
    all_contexts = read_jsonl(PROCESSED / "tatqa_contexts.jsonl")
    limit = cfg["cag"]["max_prefix_tokens"]
    out = {}
    for key, hf_id in cfg["models"].items():
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
        except Exception as e:  # noqa: BLE001  (not logged in, or no access to the model yet)
            print(f"[skipped] {key}: could not load its tokenizer ({type(e).__name__}). "
                  "Log in to Hugging Face and check access to this model.")
            continue
        sizes = {}
        for split in ("test", "dev"):
            kb = KnowledgeBase([c for c in all_contexts if c["split"] == split])
            for block in sorted(kb.blocks):
                prefix, _ = encode_prompt(tok, docs_prefix(kb.block_docs(block)), question_suffix({"question": "x"}))
                sizes[f"{split} block {block}"] = len(prefix)
        out[key] = sizes
        worst = max(sizes.values())
        print(f"{key:10s} " + "  ".join(f"{k}: {v:,}" for k, v in sizes.items()) +
              f"   -> largest {worst:,} of {limit:,} allowed: {'OK' if worst <= limit else 'TOO LARGE'}")
    return out


def hit_rates(retriever: Retriever, items: list[dict], ks=(1, 3, 5)) -> dict:
    ranks = np.array([retriever.rank(it["question"], it["block"]).index(it["context_id"]) for it in items])
    out = {f"hit@{k}": round(float((ranks < k).mean()), 3) for k in ks}
    for source in sorted({it["answer_from"] for it in items}):
        mask = np.array([it["answer_from"] == source for it in items])
        out[f"hit@3 ({source} questions)"] = round(float((ranks[mask] < 3).mean()), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    args = ap.parse_args()
    cfg = load_config()
    items = read_jsonl(PROCESSED / f"tatqa_{args.split}.jsonl")
    contexts = [c for c in read_jsonl(PROCESSED / "tatqa_contexts.jsonl") if c["split"] == args.split]
    block_size = cfg["tatqa"]["block_size"]
    print(f"A. Search check: {len(items)} {args.split} questions, {len(contexts)} report extracts "
          f"in blocks of {block_size}\n")

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    results = {}
    for name, kind in [("keyword search (TF-IDF)", "tfidf"), (f"dense search ({cfg['rag']['embedder']})", "dense")]:
        try:
            results[name] = hit_rates(Retriever(contexts, kind, cfg["rag"]["embedder"], device), items)
        except Exception as e:  # noqa: BLE001  (e.g. no internet for the embedding model)
            print(f"[skipped] {name}: {e}")
    for name, r in results.items():
        print(name)
        for k, v in r.items():
            print(f"   {k:32s} {v:.1%}")

    wanted = "dense" if cfg["rag"]["retriever"] == "dense" else "tf-idf"
    name = next((n for n in results if wanted in n.lower()), next(iter(results), None))
    if name:
        h3 = results[name]["hit@3"]
        verdict = "GOOD: search works, continue to step 3." if h3 >= 0.8 else \
                  "ACCEPTABLE: continue, but report this number as a limitation." if h3 >= 0.6 else \
                  "TOO LOW: stop here and review the dataset choice before running the models."
        print(f"\nVerdict based on {name}: the right extract is in the top 3 for {h3:.1%} of questions. {verdict}")

    print("\nB. Tokens per knowledge block (what CAG preloads)")
    sizes = block_sizes(cfg)
    limit = cfg["cag"]["max_prefix_tokens"]
    if any(v > limit for m in sizes.values() for v in m.values()):
        print("\nA block is TOO LARGE: lower tatqa.block_size in configs/experiment.yaml (for example to 21), "
              "rerun scripts/01_prepare_tatqa.py and this check, before running any model.")
    elif sizes:
        print("\nAll blocks fit. Continue to step 3.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "setup_check.json").write_text(json.dumps(
        {"split": args.split, "n_questions": len(items), "block_size": block_size, "retrieval": results,
         "block_tokens": sizes, "max_prefix_tokens": limit}, indent=2))


if __name__ == "__main__":
    main()
