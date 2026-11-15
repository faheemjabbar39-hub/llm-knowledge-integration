"""One prompt for every condition. Only the DOCUMENTS part changes between strategies:

  Base   no documents
  RAG    the top-k report extracts found by search
  CAG    the whole knowledge block (preloaded once and reused through the KV cache)
  QLoRA  the same documents as RAG, answered by the fine-tuned model

The prompt is built as  prefix (documents) + suffix (instruction + question)  so that CAG
can cache the prefix and every strategy still sees exactly the same wording.
"""
from __future__ import annotations

DOCS_HEADER = "Use the following documents to answer.\n\n"

INSTRUCTION = (
    "You are answering a question about a company's financial report.\n"
    "- If a calculation is needed, write it on one line starting with 'Calculation:'.\n"
    "- Then give the final answer on a new line starting with 'Answer:'. Give only the value: "
    "a number with its scale or unit (for example 17.7%, 2.5 million, 1,234 thousand), "
    "or the exact words from the report.\n"
    "- If there are several answers, separate them with ' ; '.\n"
    "- If the answer cannot be found, write 'Answer: I don't know'."
)


def docs_prefix(docs_text: str | None) -> str:
    return DOCS_HEADER + docs_text.strip() + "\n\n" if docs_text else ""


def question_suffix(item: dict) -> str:
    return INSTRUCTION + "\n\nQuestion: " + item["question"]


def build_user_message(item: dict, docs_text: str | None = None) -> str:
    return docs_prefix(docs_text) + question_suffix(item)


def target_text(item: dict) -> str:
    """The ideal reply, used as the QLoRA training target."""
    calc = f"Calculation: {item['derivation']}\n" if item.get("derivation") else ""
    return calc + "Answer: " + item["gold_text"]


def to_chat(user_message: str) -> list[dict]:
    # No system role: Mistral-7B-v0.3's template rejects it, so instructions live in
    # the user turn for every model (keeps prompts identical across models).
    return [{"role": "user", "content": user_message}]
