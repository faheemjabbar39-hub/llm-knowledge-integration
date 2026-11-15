"""Stage 1c: clean SEC/ESMA documents, cut them into passages, and split DOCUMENTS into train/test.

Splitting by document (not by question) is what prevents leakage: no passage used
to make a test question is ever used for training QLoRA.

Output: data/processed/domain_chunks.jsonl  (every passage, tagged with split=train|test)
Run:    python scripts/03_build_domain_corpus.py
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from kis.common import DATA, PROCESSED, load_config, write_jsonl


def html_to_text(path: Path) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for tag in soup(["script", "style", "head", "ix:header"]):
        tag.decompose()
    return soup.get_text(" ")


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"\(?\+?\d[\d\s().-]{8,}\d")


def clean(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)          # remove personal contact details
    text = _PHONE.sub("[PHONE]", text)
    text = re.sub(r"Table of Contents", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk(words: list[str], size: int, overlap: int) -> list[str]:
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def main():
    cfg = load_config()
    d = cfg["domain"]
    docs = []
    for p in sorted((DATA / "raw" / "sec").glob("*.htm*")):
        docs.append(("sec", p, html_to_text(p)))
    for p in sorted((DATA / "raw" / "esma").glob("*.pdf")):
        docs.append(("esma", p, pdf_to_text(p)))
    if not docs:
        raise SystemExit("No documents found in data/raw/sec or data/raw/esma. Run script 02 / add ESMA PDFs.")

    rng = random.Random(cfg["seed"])
    names = [p.stem for _, p, _ in docs]
    rng.shuffle(names)
    n_test = max(1, round(len(names) * d["test_doc_fraction"]))
    test_docs = set(names[:n_test])

    rows = []
    for source, p, raw in docs:
        words = clean(raw).split()
        for j, c in enumerate(chunk(words, d["chunk_words"], d["chunk_overlap_words"])):
            if len(c.split()) < 80:  # skip fragments (cover pages, signatures)
                continue
            rows.append({"chunk_id": f"{p.stem}-{j:05d}", "doc": p.stem, "source": source,
                         "split": "test" if p.stem in test_docs else "train", "text": c})
    write_jsonl(rows, PROCESSED / "domain_chunks.jsonl")
    n_te = sum(r["split"] == "test" for r in rows)
    print(f"{len(docs)} documents -> {len(rows)} passages "
          f"(test docs: {sorted(test_docs)}; test passages={n_te}, train passages={len(rows) - n_te})")


if __name__ == "__main__":
    main()
