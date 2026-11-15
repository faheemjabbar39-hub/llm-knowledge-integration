"""A tiny random Llama-style model and tokenizer, built in memory (no download).
Used only to test the pipeline mechanics (prompt building, KV-cache reuse, result files)."""
from __future__ import annotations

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

TEMPLATE = ("{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] | trim }}<|end|>\n"
            "{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}")


def tiny_model_and_tokenizer(seed: int = 0):
    torch.manual_seed(seed)
    specials = ["<pad>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|end|>"]
    alphabet = [chr(i) for i in range(32, 127)] + ["\n", "’", "•", "€", "£", "−", "–"]
    vocab = {t: i for i, t in enumerate(specials + ["<unk>"] + alphabet)}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Split("", "isolated")      # one token per character
    t.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  unk_token="<unk>", additional_special_tokens=specials[3:])
    tok.chat_template = TEMPLATE
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=65536,
                      bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return LlamaForCausalLM(cfg).eval(), tok
