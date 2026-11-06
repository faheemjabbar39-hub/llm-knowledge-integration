"""Stage 2+: run one condition (model x strategy) over the TEST sets and record per-item results.

Examples
  Quick smoke test (20 items per dataset):
    python scripts/05_run_eval.py --model llama8b --strategy base --limit 20
  Full baseline:
    python scripts/05_run_eval.py --model llama8b --strategy base
  QLoRA (after training an adapter):
    python scripts/05_run_eval.py --model llama8b --strategy qlora --adapter adapters/llama8b_r16_s42

Output: results/raw/<model>__<strategy>[__tag].csv (one row per item) + a .meta.json summary.
The script resumes: rerunning skips items already in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from tqdm import tqdm

from kis.common import PROCESSED, RESULTS, load_config, read_jsonl, set_seed
from kis.metrics import score_item
from kis.models import generate, load_model, peak_memory_gb, reset_peak_memory
from kis.prompts import build_user_message, to_chat
from kis.strategies import make_strategy

FIELDS = ["item_id", "dataset", "model", "strategy", "tag", "seed", "question", "gold", "prediction",
          "pred_choice", "correct", "contains", "hallucinated", "latency_ms", "prompt_tokens",
          "output_tokens", "device", "quant"]


def load_test_items(datasets: list[str] | None, limit: int | None) -> list[dict]:
    files = sorted(PROCESSED.glob("*_test.jsonl"))
    items = []
    for f in files:
        name = f.name.removesuffix("_test.jsonl")
        if datasets and name not in datasets:
            continue
        rows = read_jsonl(f)
        items += rows[:limit] if limit else rows
    if not items:
        raise SystemExit("No test files found in data/processed. Run scripts 01 (and 02-04 for domain) first.")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key from configs/experiment.yaml (llama8b, llama70b, mistral7b)")
    ap.add_argument("--strategy", default="base", choices=["base", "qlora", "rag", "cag"])
    ap.add_argument("--adapter", default=None, help="path to QLoRA adapter (qlora only)")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset, e.g. nq truthfulqa domain")
    ap.add_argument("--limit", type=int, default=None, help="items per dataset (for quick tests)")
    ap.add_argument("--tag", default="", help="label for sweep runs, e.g. r16 or k5")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    strategy = make_strategy(args.strategy, args.adapter)
    items = load_test_items(args.datasets, args.limit)
    random.Random(cfg["seed"]).shuffle(items)  # randomised item order, fixed seed

    stem = f"{args.model}__{args.strategy}" + (f"__{args.tag}" if args.tag else "")
    out_csv = RESULTS / "raw" / f"{stem}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            done = {r["item_id"] for r in csv.DictReader(f)}
    todo = [it for it in items if it["id"] not in done]
    print(f"{stem}: {len(items)} items, {len(done)} already done, {len(todo)} to run")
    if not todo:
        return

    model, tok, device = load_model(cfg["models"][args.model], cfg["load_in_4bit"], strategy.adapter_path)
    quant = "nf4" if (cfg["load_in_4bit"] and device == "cuda") else "fp16"
    reset_peak_memory(device)
    t_start = time.time()

    new_file = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for it in tqdm(todo):
            msg = to_chat(build_user_message(it, strategy.context_for(it)))
            g = generate(model, tok, device, msg, cfg["generation"]["max_new_tokens"])
            s = score_item(it, g.text)
            w.writerow({
                "item_id": it["id"], "dataset": it["dataset"], "model": args.model,
                "strategy": args.strategy, "tag": args.tag, "seed": cfg["seed"],
                "question": it["question"],
                "gold": it.get("answer_letter") or " || ".join(it["answers"]),
                "prediction": g.text, **s, "latency_ms": round(g.latency_ms, 2),
                "prompt_tokens": g.prompt_tokens, "output_tokens": g.output_tokens,
                "device": device, "quant": quant,
            })
            f.flush()

    meta = {"model": args.model, "hf_id": cfg["models"][args.model], "strategy": args.strategy,
            "tag": args.tag, "device": device, "quant": quant, "n_items_this_run": len(todo),
            "wall_clock_s": round(time.time() - t_start, 1), "peak_gpu_gb": round(peak_memory_gb(device), 3)}
    Path(str(out_csv).replace(".csv", ".meta.json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
