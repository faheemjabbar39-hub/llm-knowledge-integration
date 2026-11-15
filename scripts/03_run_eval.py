"""Step 3: run one model under one or more strategies and record per-question results.

Examples
  Smoke test (30 questions):
    python scripts/03_run_eval.py --model llama8b --strategy base rag cag --limit 30
  Full run of the three strategies that need no training (the model is loaded once):
    python scripts/03_run_eval.py --model llama8b --strategy base rag cag
  QLoRA (after training an adapter with scripts/04_train_qlora.py):
    python scripts/03_run_eval.py --model llama8b --strategy qlora --adapter adapters/llama8b_r16_s42
  QLoRA without documents (extra check):
    python scripts/03_run_eval.py --model llama8b --strategy qlora --adapter adapters/llama8b_r16_s42 --no-docs --tag nodocs

Output: results/raw/<model>__<strategy>[__tag].csv (one row per question) and a .meta.json summary.
Each model and strategy has its own file, so nothing is overwritten. Rerunning a command
skips the questions already answered and continues where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from kis.common import PROCESSED, RESULTS, ROOT, load_config, read_jsonl, set_seed
from kis.models import (PrefixCache, encode_prompt, generate, load_model, peak_memory_gb,
                        reset_peak_memory)
from kis.prompts import docs_prefix, question_suffix
from kis.retrieval import Retriever
from kis.strategies import STRATEGIES, KnowledgeBase
from kis.tatqa import score_tatqa

FIELDS = ["item_id", "dataset", "split", "block", "answer_type", "answer_from", "model", "strategy", "tag",
          "seed", "question", "gold", "prediction", "final_answer", "correct", "contains", "f1", "abstained",
          "hallucinated", "retrieval_hit", "retrieved", "latency_ms", "retrieval_ms", "prompt_tokens",
          "output_tokens", "device", "quant"]


def ordered_items(split: str, seed: int, limit: int | None) -> list[dict]:
    path = PROCESSED / f"tatqa_{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path.name} not found. Run scripts/01_prepare_tatqa.py first.")
    items = read_jsonl(path)
    random.Random(seed).shuffle(items)              # random order, fixed seed
    items.sort(key=lambda it: it["block"])          # keep each block together (one CAG cache per block)
    return items[:limit] if limit else items


def run_condition(args, cfg, strategy, items, kb, retriever, model, tok, device, out_dir):
    use_docs = strategy in ("rag", "qlora") and not args.no_docs
    stem = f"{args.model}__{strategy}" + (f"__{args.tag}" if args.tag else "")
    out_csv = out_dir / f"{stem}.csv"
    done = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            done = {r["item_id"] for r in csv.DictReader(f)}
    todo = [it for it in items if it["id"] not in done]
    print(f"\n=== {stem}: {len(items)} questions, {len(done)} already done, {len(todo)} to run")
    if not todo:
        return

    quant = "nf4" if (cfg["load_in_4bit"] and device == "cuda") else "fp16"
    max_new = cfg["generation"]["max_new_tokens"]
    top_k = args.top_k or cfg["rag"]["top_k"]
    limit_tokens = min(cfg["cag"]["max_prefix_tokens"],
                       getattr(model.config, "max_position_embeddings", 10**9) - max_new - 400)
    reset_peak_memory(device)
    t_start = time.time()
    cache, cache_block, cache_prefix, cache_build = None, None, None, {}

    new_file = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for it in tqdm(todo, desc=stem):
            retrieved, retrieval_ms, hit, prefix_cache = [], 0.0, None, None
            suffix = question_suffix(it)
            if use_docs:
                retrieved, retrieval_ms = retriever.retrieve(it["question"], it["block"], top_k)
                hit = int(it["context_id"] in retrieved)
                prefix_ids, suffix_ids = encode_prompt(tok, docs_prefix(kb.docs(retrieved)), suffix)
            elif strategy == "cag":
                prefix_ids, suffix_ids = encode_prompt(tok, docs_prefix(kb.block_docs(it["block"])), suffix)
                if cache_block != it["block"]:      # first question of a block: read its documents once
                    if len(prefix_ids) > limit_tokens:
                        raise SystemExit(f"Block {it['block']} is {len(prefix_ids):,} tokens, above the limit of "
                                         f"{limit_tokens:,} for this model. Lower tatqa.block_size in "
                                         "configs/experiment.yaml and rerun scripts/01_prepare_tatqa.py.")
                    cache = None
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    try:
                        cache = PrefixCache(model, device, prefix_ids, cfg["cag"]["prefill_chunk_tokens"])
                    except torch.OutOfMemoryError:
                        raise SystemExit(f"The GPU ran out of memory preloading block {it['block']} "
                                         f"({len(prefix_ids):,} tokens). Lower tatqa.block_size in "
                                         "configs/experiment.yaml, rerun scripts/01_prepare_tatqa.py, and "
                                         "rerun every strategy so all of them use the same blocks.") from None
                    cache_block, cache_prefix = it["block"], prefix_ids
                    cache_build[str(it["block"])] = round(cache.build_s, 2)
                    print(f"  block {it['block']}: {len(prefix_ids):,} tokens preloaded in {cache.build_s:.1f}s")
                if prefix_ids != cache_prefix:
                    raise RuntimeError("document prefix changed within a block")
                prefix_cache = cache
            else:                                   # base, or qlora --no-docs
                prefix_ids, suffix_ids = encode_prompt(tok, "", suffix)

            g = generate(model, tok, device, prefix_ids, suffix_ids, max_new, prefix_cache)
            s = score_tatqa(it, g.text)
            w.writerow({
                "item_id": it["id"], "dataset": it["dataset"], "split": it["split"], "block": it["block"],
                "answer_type": it["answer_type"], "answer_from": it["answer_from"], "model": args.model,
                "strategy": strategy, "tag": args.tag, "seed": cfg["seed"], "question": it["question"],
                "gold": it["gold_text"], "prediction": g.text, **s, "retrieval_hit": hit,
                "retrieved": " ".join(retrieved), "latency_ms": round(g.latency_ms + retrieval_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2), "prompt_tokens": g.prompt_tokens,
                "output_tokens": g.output_tokens, "device": device, "quant": quant,
            })
            f.flush()

    # one-off preparation cost, reported separately from the per-question cost
    setup = {}
    if use_docs:
        setup["index_build_s"] = round(retriever.build_s, 2)
    if strategy == "cag":
        setup["cache_build_s_by_block"] = cache_build
    if strategy == "qlora" and args.adapter:
        tm = Path(args.adapter) / "train_meta.json"
        if tm.exists():
            setup["qlora_train_s"] = json.loads(tm.read_text()).get("train_s")

    meta_path = out_csv.with_suffix(".meta.json")
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if strategy == "cag":
        setup["cache_build_s_by_block"] = {**old.get("setup", {}).get("cache_build_s_by_block", {}), **cache_build}
    meta = {"model": args.model, "hf_id": cfg["models"][args.model], "strategy": strategy, "tag": args.tag,
            "split": args.split, "device": device, "quant": quant, "uses_documents": use_docs or strategy == "cag",
            "top_k": top_k if use_docs else None, "retriever": retriever.kind if use_docs else None,
            "adapter": args.adapter, "n_items_this_run": len(todo),
            "wall_clock_s": round(old.get("wall_clock_s", 0) + time.time() - t_start, 1),
            "peak_gpu_gb": round(max(old.get("peak_gpu_gb") or 0, peak_memory_gb(device)), 3)
            if device == "cuda" else None, "setup": setup}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main(model_and_tok=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key from configs/experiment.yaml (llama8b, mistral7b, qwen7b)")
    ap.add_argument("--strategy", nargs="+", default=["base"], choices=STRATEGIES)
    ap.add_argument("--adapter", default=None, help="path to the trained QLoRA adapter (qlora only)")
    ap.add_argument("--no-docs", action="store_true", help="qlora only: answer without documents")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    ap.add_argument("--tag", default="", help="label for an extra run, e.g. nodocs or k5")
    ap.add_argument("--top-k", type=int, default=None, help="override rag.top_k")
    ap.add_argument("--retriever", default=None, choices=["dense", "tfidf"], help="override rag.retriever")
    args = ap.parse_args()

    if "qlora" in args.strategy:
        if len(args.strategy) > 1:
            raise SystemExit("Run qlora on its own: it needs the fine-tuned model, the others need the original.")
        if not args.adapter:
            raise SystemExit("--adapter is required for the qlora strategy")
        if not (ROOT / args.adapter).exists() and not Path(args.adapter).exists():
            raise SystemExit(f"Adapter {args.adapter} not found. Train it with scripts/04_train_qlora.py first.")
    if args.no_docs and not args.tag:
        args.tag = "nodocs"

    cfg = load_config()
    set_seed(cfg["seed"])
    items = ordered_items(args.split, cfg["seed"], args.limit)
    kb = KnowledgeBase([c for c in read_jsonl(PROCESSED / "tatqa_contexts.jsonl") if c["split"] == args.split])
    out_dir = RESULTS / ("raw" if args.split == "test" else args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_and_tok is None:
        if cfg["load_in_4bit"] and not torch.cuda.is_available():
            raise SystemExit("No NVIDIA GPU found, so the model cannot be loaded in 4-bit.\n"
                             "In Colab: Runtime -> Change runtime type -> T4 GPU -> Save. The session restarts,\n"
                             "so run Setup A, B, C and Step 1 again, then this cell.")
        model, tok, device = load_model(cfg["models"][args.model], cfg["load_in_4bit"], args.adapter)
    else:                                           # used by the tests
        model, tok, device = model_and_tok

    retriever = None
    if any(s in ("rag", "qlora") for s in args.strategy) and not args.no_docs:
        kind = args.retriever or cfg["rag"]["retriever"]
        retriever = Retriever(kb.contexts, kind, cfg["rag"]["embedder"], device if device != "mps" else "cpu")
        print(f"search index built in {retriever.build_s:.1f}s ({kind})")

    for strategy in args.strategy:
        run_condition(args, cfg, strategy, items, kb, retriever, model, tok, device, out_dir)


if __name__ == "__main__":
    main()
