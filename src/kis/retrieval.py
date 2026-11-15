"""Retrieval for RAG: find the report extracts most relevant to a question.

Each extract is cut into small pieces (table rows in groups, paragraphs). The question is
compared with every piece; an extract's score is its best piece. The top-k extracts are
returned whole. Search is exact cosine similarity (the same result a FAISS flat index gives).
"""
from __future__ import annotations

import time

import numpy as np


def chunk_text(text: str, rows_per_chunk: int = 10, max_words: int = 120) -> list[str]:
    """Split one serialised extract into retrievable pieces."""
    table_part, _, text_part = text.partition("\n\nText:\n")
    rows = table_part.removeprefix("Table:\n").split("\n")
    header, body = rows[0], rows[1:] or [""]
    chunks = [header + "\n" + "\n".join(body[i:i + rows_per_chunk]) for i in range(0, len(body), rows_per_chunk)]
    buf: list[str] = []
    for para in [p for p in text_part.split("\n") if p.strip()]:
        buf.append(para)
        if sum(len(b.split()) for b in buf) >= max_words:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks


class DenseEmbedder:
    """Sentence-BERT embeddings (all-mpnet-base-v2 by default)."""

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def fit(self, texts: list[str]) -> None:  # nothing to fit
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                            batch_size=32), dtype=np.float32)


class TfidfEmbedder:
    """Keyword (TF-IDF) vectors: a simple baseline that needs no model download."""

    def __init__(self, *_, **__):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))

    def fit(self, texts: list[str]) -> None:
        self.vec.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.vec.transform(texts).todense(), dtype=np.float32)  # rows are L2-normalised


def make_embedder(kind: str, model_name: str = "", device: str | None = None):
    if kind == "dense":
        return DenseEmbedder(model_name, device)
    if kind == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"unknown retriever '{kind}' (use dense or tfidf)")


class Retriever:
    """Searches inside one knowledge block (the same extracts CAG preloads for that block)."""

    def __init__(self, contexts: list[dict], kind: str, model_name: str = "", device: str | None = None):
        t0 = time.perf_counter()
        self.kind = kind
        self.by_block: dict[int, dict] = {}
        dense = make_embedder(kind, model_name, device) if kind == "dense" else None
        for block in sorted({c["block"] for c in contexts}):
            ctxs = [c for c in contexts if c["block"] == block]
            pieces, parent = [], []
            for i, c in enumerate(ctxs):
                for ch in chunk_text(c["text"]):
                    pieces.append(ch)
                    parent.append(i)
            emb = dense or make_embedder(kind)          # TF-IDF vocabulary is fitted per block
            emb.fit(pieces)
            self.by_block[block] = {"ids": [c["context_id"] for c in ctxs], "emb": emb,
                                    "matrix": emb.encode(pieces), "parent": np.asarray(parent)}
        self.build_s = time.perf_counter() - t0

    def rank(self, question: str, block: int) -> list[str]:
        """All extract ids in the block, best first."""
        b = self.by_block[block]
        sims = b["matrix"] @ b["emb"].encode([question])[0]
        best = np.full(len(b["ids"]), -np.inf)
        np.maximum.at(best, b["parent"], sims)
        return [b["ids"][i] for i in np.argsort(-best, kind="stable")]

    def retrieve(self, question: str, block: int, k: int) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        ids = self.rank(question, block)[:k]
        return ids, (time.perf_counter() - t0) * 1000
