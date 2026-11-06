"""Model layer: loads a model identically for every strategy and times each generation."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(hf_id: str, load_in_4bit: bool = True, adapter_path: str | None = None):
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {}
    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = "auto"
    else:
        if load_in_4bit:
            print(f"[warn] 4-bit needs an NVIDIA GPU; loading in float16 on {device}. "
                  "Do NOT mix these numbers with 4-bit GPU runs in the final analysis.")
        kwargs["torch_dtype"] = torch.float16 if device != "cpu" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    if adapter_path:  # QLoRA condition: attach trained adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok, device


@dataclass
class GenResult:
    text: str
    latency_ms: float
    prompt_tokens: int
    output_tokens: int


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def generate(model, tok, device: str, messages: list[dict], max_new_tokens: int = 64,
             past_key_values=None) -> GenResult:
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    input_ids = enc["input_ids"].to(model.device)
    _sync(device)
    t0 = time.perf_counter()
    out = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy = temperature 0
        pad_token_id=tok.pad_token_id,
        past_key_values=past_key_values,
    )
    _sync(device)
    latency = (time.perf_counter() - t0) * 1000
    new = out[0, input_ids.shape[1]:]
    return GenResult(tok.decode(new, skip_special_tokens=True).strip(), latency,
                     input_ids.shape[1], int(new.shape[0]))


def reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return float("nan")
