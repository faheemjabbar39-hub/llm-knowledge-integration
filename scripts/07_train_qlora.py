"""Stage 3: QLoRA fine-tuning on the TRAIN split only (needs an NVIDIA GPU).

Trains on data/processed/*_train.jsonl (domain + benchmarks). TruthfulQA has no
train file, so it is never trained on. Prompts use exactly the same template as
evaluation, and the loss is computed on the answer tokens only.

Run (one rank, one seed):
  python scripts/07_train_qlora.py --model llama8b --rank 16 --seed 42
Sweep from the proposal: --rank 8 / 16 / 32, seeds 42 / 43 / 44.
Then evaluate:
  python scripts/05_run_eval.py --model llama8b --strategy qlora --adapter adapters/llama8b_r16_s42 --tag r16
"""
from __future__ import annotations

import argparse

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments)

from kis.common import PROCESSED, ROOT, load_config, read_jsonl, set_seed
from kis.prompts import build_user_message, to_chat


def build_examples(tok, max_len: int) -> Dataset:
    rows = []
    for f in sorted(PROCESSED.glob("*_train.jsonl")):
        rows += read_jsonl(f)
    if not rows:
        raise SystemExit("No *_train.jsonl files. Run scripts 01-04 first.")
    feats = []
    for it in rows:
        answer = it["answer_letter"] if it["type"] == "mc" else it["answers"][0]
        prompt_ids = tok.apply_chat_template(to_chat(build_user_message(it)), add_generation_prompt=True,
                                             tokenize=True, return_dict=True)["input_ids"]
        ans_ids = tok(" " + answer + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + ans_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + ans_ids)[:max_len]   # learn the answer, not the prompt
        feats.append({"input_ids": ids, "labels": labels})
    print(f"{len(feats)} training examples from {len(rows)} rows")
    return Dataset.from_list(feats)


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA needs an NVIDIA GPU (bitsandbytes). Use Colab/Kaggle/institutional GPU.")

    cfg = load_config()
    set_seed(args.seed)
    hf_id = cfg["models"][args.model]
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.pad_token = tok.pad_token or tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    out = ROOT / "adapters" / f"{args.model}_r{args.rank}_s{args.seed}"
    trainer = Trainer(
        model=model,
        train_dataset=build_examples(tok, args.max_len),
        data_collator=collate(tok.pad_token_id),
        args=TrainingArguments(
            output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
            lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10, save_strategy="no",
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            optim="paged_adamw_8bit", seed=args.seed, report_to="none", remove_unused_columns=False),
    )
    trainer.train()
    model.save_pretrained(out)
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()
