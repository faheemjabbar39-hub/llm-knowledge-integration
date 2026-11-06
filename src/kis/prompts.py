"""One prompt template for all conditions. Only the CONTEXT block changes between strategies:
Base = no context, RAG = retrieved passages, CAG = preloaded corpus (via KV cache), QLoRA = no context."""
from __future__ import annotations

import string

OPEN_INSTRUCTION = (
    "Answer the question with a short phrase only (a name, number, date or a few words). "
    "Do not explain. If you do not know, answer exactly: I don't know."
)
MC_INSTRUCTION = "Choose the correct option. Reply with the option letter only."


def build_user_message(item: dict, context: str | None = None) -> str:
    parts = []
    if context:
        parts.append("Use the following documents to answer.\n\n" + context.strip())
    parts.append(MC_INSTRUCTION if item["type"] == "mc" else OPEN_INSTRUCTION)
    q = f"Question: {item['question']}"
    if item["type"] == "mc":
        letters = string.ascii_uppercase
        q += "\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item["choices"]))
    parts.append(q + "\nAnswer:")
    return "\n\n".join(parts)


def to_chat(user_message: str) -> list[dict]:
    # No system role: Mistral-7B-v0.3's template rejects it, so instructions live in
    # the user turn for every model (keeps prompts identical across models).
    return [{"role": "user", "content": user_message}]
