"""QLoRA fine-tuning: build training examples and train small LoRA adapters.

Training examples look exactly like the test prompts of the RAG condition: the question,
its own report extract mixed with other extracts (so the model learns to pick the right
one), and the ideal reply (calculation line + answer) as the target.
"""
from __future__ import annotations

import math
import random
import time

import torch

from kis.models import encode_prompt
from kis.prompts import docs_prefix, question_suffix, target_text
from kis.tatqa import format_docs


def build_examples(tok, items: list[dict], contexts: list[dict], n_docs: int, max_len: int, seed: int):
    """Returns ([{input_ids, labels}], stats). The loss is computed on the reply only, not on the prompt."""
    rng = random.Random(seed)
    text = {c["context_id"]: c["text"] for c in contexts}
    ids = list(text)
    examples, stats = [], {"total": len(items), "kept": 0, "fewer_docs": 0, "skipped_too_long": 0}
    for it in items:
        others = rng.sample([i for i in ids if i != it["context_id"]], min(n_docs - 1, len(ids) - 1))
        target = tok(target_text(it), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        for doc_ids in (others + [it["context_id"]], [it["context_id"]]):   # fall back to the gold extract only
            docs = list(doc_ids)
            rng.shuffle(docs)
            prefix, suffix = encode_prompt(tok, docs_prefix(format_docs([text[d] for d in docs])),
                                           question_suffix(it))
            prompt = prefix + suffix
            if len(prompt) + len(target) <= max_len:
                examples.append({"input_ids": prompt + target, "labels": [-100] * len(prompt) + target})
                stats["kept"] += 1
                stats["fewer_docs"] += int(len(docs) < n_docs)
                break
        else:
            stats["skipped_too_long"] += 1
    return examples, stats


def add_lora(model, rank: int, alpha: int, dropout: float, target_modules: list[str]):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                                            target_modules=list(target_modules), task_type="CAUSAL_LM"))


def train(model, examples: list[dict], epochs: float = 1, lr: float = 2e-4, grad_accum: int = 8,
          seed: int = 42, log_every: int = 20, autocast_dtype=None) -> dict:
    """A plain training loop (batch size 1 with gradient accumulation). Returns a small log."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n_steps = max(1, math.ceil(len(examples) * epochs))
    n_updates = max(1, math.ceil(n_steps / grad_accum))
    warmup = max(1, int(0.05 * n_updates))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: (u + 1) / warmup if u < warmup
        else 0.5 * (1 + math.cos(math.pi * (u - warmup) / max(1, n_updates - warmup))))
    device = next(model.parameters()).device
    model.train()
    order, losses, t0 = [], [], time.time()
    while len(order) < n_steps:
        epoch = list(range(len(examples)))
        rng.shuffle(epoch)
        order += epoch
    for step, idx in enumerate(order[:n_steps], 1):
        ex = examples[idx]
        ids = torch.tensor([ex["input_ids"]], device=device)
        labels = torch.tensor([ex["labels"]], device=device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            loss = model(input_ids=ids, labels=labels).loss
        (loss / grad_accum).backward()
        losses.append(loss.item())
        if step % grad_accum == 0 or step == n_steps:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if step % log_every == 0 or step == n_steps:
            recent = sum(losses[-log_every:]) / len(losses[-log_every:])
            mins = (time.time() - t0) / 60
            print(f"  step {step}/{n_steps}  loss {recent:.3f}  {mins:.1f} min elapsed, "
                  f"about {mins / step * (n_steps - step):.0f} min left", flush=True)
    model.eval()
    k = max(1, len(losses) // 10)
    return {"steps": n_steps, "train_s": round(time.time() - t0, 1),
            "loss_start": round(sum(losses[:k]) / k, 4), "loss_end": round(sum(losses[-k:]) / k, 4)}
