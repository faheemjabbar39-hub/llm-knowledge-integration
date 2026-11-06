"""Knowledge layer: the ONLY thing that changes between conditions.

Each strategy returns the context (if any) to put in the prompt for a given item.
Base and QLoRA add no context (QLoRA's knowledge is inside the adapter weights).
RAG and CAG are filled in at later stages of the project.
"""
from __future__ import annotations


class Strategy:
    name = "base"
    adapter_path: str | None = None

    def context_for(self, item: dict) -> str | None:
        return None


class Base(Strategy):
    name = "base"


class QLoRA(Strategy):
    name = "qlora"

    def __init__(self, adapter_path: str):
        self.adapter_path = adapter_path


class RAG(Strategy):  # Stage 4
    name = "rag"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RAG is built in stage 4 (FAISS index + top-k retrieval).")


class CAG(Strategy):  # Stage 5
    name = "cag"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CAG is built in stage 5 (preloaded corpus + reused KV cache).")


def make_strategy(name: str, adapter_path: str | None = None) -> Strategy:
    if name == "base":
        return Base()
    if name == "qlora":
        if not adapter_path:
            raise ValueError("--adapter is required for the qlora strategy")
        return QLoRA(adapter_path)
    if name == "rag":
        return RAG()
    if name == "cag":
        return CAG()
    raise ValueError(f"unknown strategy {name}")
