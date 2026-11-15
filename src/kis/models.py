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
        kwargs["dtype"] = torch.float16 if device != "cpu" else torch.float32

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


def encode_prompt(tok, prefix_text: str, suffix_text: str) -> tuple[list[int], list[int]]:
    """Token ids of the chat prompt, split at the point where the documents end.

    The user message is prefix_text (documents, may be empty) + suffix_text (instruction and
    question). Every strategy is tokenised this way, so prompts differ only in their documents.
    """
    rendered = tok.apply_chat_template([{"role": "user", "content": prefix_text + suffix_text}],
                                       tokenize=False, add_generation_prompt=True)
    cut = rendered.find(suffix_text[:60])
    if cut < 0:
        raise ValueError("chat template changed the prompt text; cannot split it for caching")
    prefix_ids = tok(rendered[:cut], add_special_tokens=False)["input_ids"]
    suffix_ids = tok(rendered[cut:], add_special_tokens=False)["input_ids"]
    return list(prefix_ids), list(suffix_ids)


class PrefixCache:
    """CAG: read the documents once, keep the model's key-value (KV) cache, reuse it for every question."""

    def __init__(self, model, device: str, prefix_ids: list[int], chunk_tokens: int = 1024):
        from transformers import DynamicCache
        self.n = len(prefix_ids)
        ids = torch.tensor([prefix_ids], device=model.device)
        cache = DynamicCache()
        _sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for s in range(0, self.n, chunk_tokens):     # in pieces, so GPU memory stays low
                out = model(input_ids=ids[:, s:s + chunk_tokens], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
        _sync(device)
        self.cache = cache
        self.build_s = time.perf_counter() - t0

    def reset(self) -> bool:
        """Cut the cache back to the documents only (drops the last question and answer)."""
        extra = self.cache.get_seq_length() - self.n
        if extra > 0:
            self.cache.crop(-extra)      # a negative value removes that many tokens from the end
        return self.cache.get_seq_length() == self.n


@torch.inference_mode()
def generate(model, tok, device: str, prefix_ids: list[int], suffix_ids: list[int],
             max_new_tokens: int = 96, prefix_cache: PrefixCache | None = None) -> GenResult:
    input_ids = torch.tensor([prefix_ids + suffix_ids], device=model.device)
    kwargs = {"past_key_values": prefix_cache.cache} if prefix_cache is not None else {}
    _sync(device)
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy = temperature 0
        pad_token_id=tok.pad_token_id,
        **kwargs,
    )
    _sync(device)
    latency = (time.perf_counter() - t0) * 1000
    if prefix_cache is not None and not prefix_cache.reset():
        raise RuntimeError("KV cache could not be reset to the document prefix")
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
