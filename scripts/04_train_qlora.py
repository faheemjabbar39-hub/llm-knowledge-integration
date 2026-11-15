"""Step 4: QLoRA fine-tuning on the TRAIN questions only (needs an NVIDIA GPU).

The model is loaded in 4-bit, small LoRA adapters are added and trained on TAT-QA train
questions with their report extracts. Test and dev questions are never seen.

Run:   python scripts/04_train_qlora.py --model llama8b
Then:  python scripts/03_run_eval.py --model llama8b --strategy qlora --adapter adapters/llama8b_r16_s42

Output: adapters/<model>_r<rank>_s<seed>/ (the adapter and train_meta.json)
"""
from __future__ import annotations

import argparse
import json

import torch

from kis.autosave import autosave
from kis.common import PROCESSED, ROOT, load_config, read_jsonl, set_seed
from kis.qlora import add_lora, build_examples, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-examples", type=int, default=None, help="use fewer training questions (faster)")
    ap.add_argument("--force", action="store_true", help="train again even if the adapter exists")
    args = ap.parse_args()

    cfg = load_config()
    q = cfg["qlora"]
    rank, seed = args.rank or q["rank"], args.seed or cfg["seed"]
    out = ROOT / "adapters" / f"{args.model}_r{rank}_s{seed}"
    if (out / "train_meta.json").exists() and not args.force:
        print(f"Adapter already trained: {out} (use --force to train again)")
        return
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA needs an NVIDIA GPU. In Colab: Runtime -> Change runtime type -> T4 GPU.")

    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    set_seed(seed)
    hf_id = cfg["models"][args.model]
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.pad_token = tok.pad_token or tok.eos_token

    items = read_jsonl(PROCESSED / "tatqa_train.jsonl")
    if args.max_examples:
        items = items[:args.max_examples]
    contexts = [c for c in read_jsonl(PROCESSED / "tatqa_contexts.jsonl") if c["split"] == "train"]
    examples, stats = build_examples(tok, items, contexts, cfg["rag"]["top_k"], q["max_len"], seed)
    print(f"training examples: {stats}")

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model)      # also turns on gradient checkpointing
    model = add_lora(model, rank, q["alpha"], q["dropout"], q["target_modules"])
    model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    log = train(model, examples, q["epochs"], q["learning_rate"], seed=seed, autocast_dtype=torch.bfloat16)
    model.save_pretrained(out)
    meta = {"model": args.model, "hf_id": hf_id, "rank": rank, "alpha": q["alpha"], "seed": seed,
            "examples": stats, **log, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"adapter saved to {out}")
    autosave(f"QLoRA adapter {out.name} trained")


if __name__ == "__main__":
    main()
