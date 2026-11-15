"""Knowledge layer: the ONLY thing that changes between conditions.

  base   question only
  rag    question + the top-k report extracts found by search in the question's block
  cag    question + the whole block, preloaded once and reused through the KV cache
  qlora  fine-tuned model + the same documents as rag (or none, with --no-docs)
"""
from __future__ import annotations

from kis.tatqa import format_docs

STRATEGIES = ["base", "rag", "cag", "qlora"]


class KnowledgeBase:
    """The report extracts of one split, grouped into blocks."""

    def __init__(self, contexts: list[dict]):
        self.contexts = contexts
        self.text = {c["context_id"]: c["text"] for c in contexts}
        self.blocks: dict[int, list[str]] = {}
        for c in contexts:
            self.blocks.setdefault(c["block"], []).append(c["context_id"])

    def docs(self, context_ids: list[str]) -> str:
        return format_docs([self.text[i] for i in context_ids])

    def block_docs(self, block: int) -> str:
        return self.docs(self.blocks[block])
