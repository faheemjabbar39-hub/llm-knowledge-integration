"""Stage 1a: download the four public benchmarks and split each into TRAIN and TEST.

Train and test come from different official splits of each dataset, so a test
question can never appear in training. TruthfulQA is test-only (it is the
hallucination probe and must never be trained on).

Output: data/processed/<dataset>_train.jsonl and data/processed/<dataset>_test.jsonl
Run:    python scripts/01_prepare_benchmarks.py
"""
from __future__ import annotations

import random
import string

from datasets import load_dataset

from kis.common import PROCESSED, load_config, write_jsonl

LETTERS = string.ascii_uppercase


def _load(ids: list[str], config: str | None, split: str):
    """Try several Hub ids (datasets get renamed over time)."""
    err = None
    for hf_id in ids:
        try:
            return load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"Could not load any of {ids} ({config}, {split}): {err}")


def _sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def hotpotqa(cfg, rng, n_train, n_test):
    ids = ["hotpotqa/hotpot_qa", "hotpot_qa"]

    def conv(ex, split):
        docs = [f"{t}: {''.join(s)}" for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
        return {"id": f"hotpotqa-{ex['id']}", "dataset": "hotpotqa", "split": split, "type": "open",
                "question": ex["question"], "answers": [ex["answer"]],
                "context_docs": docs, "gold_titles": sorted(set(ex["supporting_facts"]["title"]))}

    tr = _load(ids, cfg["config"], cfg["train_split"]).shuffle(seed=rng.randint(0, 10**6)).select(range(n_train))
    te = _load(ids, cfg["config"], cfg["test_split"]).shuffle(seed=rng.randint(0, 10**6)).select(range(n_test))
    return [conv(e, "train") for e in tr], [conv(e, "test") for e in te]


def nq(cfg, rng, n_train, n_test):
    ids = [cfg["hf_id"], "nq_open"]

    def conv(i, ex, split):
        return {"id": f"nq-{split}-{i}", "dataset": "nq", "split": split, "type": "open",
                "question": ex["question"], "answers": list(ex["answer"])}

    tr = _load(ids, None, cfg["train_split"]).shuffle(seed=rng.randint(0, 10**6)).select(range(n_train))
    te = _load(ids, None, cfg["test_split"]).shuffle(seed=rng.randint(0, 10**6)).select(range(n_test))
    return [conv(i, e, "train") for i, e in enumerate(tr)], [conv(i, e, "test") for i, e in enumerate(te)]


def mmlu_fin(cfg, rng, n_train, n_test):
    def conv(i, ex, subject, split):
        return {"id": f"mmlu-{subject}-{split}-{i}", "dataset": "mmlu_fin", "split": split, "type": "mc",
                "subject": subject, "question": ex["question"], "choices": list(ex["choices"]),
                "answer_letter": LETTERS[int(ex["answer"])]}

    train, test = [], []
    for subj in cfg["subjects"]:
        for sp in cfg["train_split"]:
            train += [conv(i, e, subj, sp) for i, e in enumerate(_load([cfg["hf_id"]], subj, sp))]
        test += [conv(i, e, subj, "test") for i, e in enumerate(_load([cfg["hf_id"]], subj, cfg["test_split"]))]
    test_q = {r["question"] for r in test}
    train = [r for r in train if r["question"] not in test_q]  # belt-and-braces de-duplication
    return _sample(train, n_train, rng), _sample(test, n_test, rng)


def truthfulqa(cfg, rng, n_train, n_test):
    ds = _load([cfg["hf_id"], "truthful_qa"], cfg["config"], cfg["test_split"])
    rows = []
    for i, ex in enumerate(ds):
        choices = list(ex["mc1_targets"]["choices"])
        correct = choices[list(ex["mc1_targets"]["labels"]).index(1)]
        rng.shuffle(choices)  # the correct option is always first in the raw data; shuffle to remove position bias
        rows.append({"id": f"truthfulqa-{i}", "dataset": "truthfulqa", "split": "test", "type": "mc",
                     "question": ex["question"], "choices": choices,
                     "answer_letter": LETTERS[choices.index(correct)]})
    return [], _sample(rows, n_test, rng)


BUILDERS = {"hotpotqa": hotpotqa, "nq": nq, "mmlu_fin": mmlu_fin, "truthfulqa": truthfulqa}


def main():
    cfg = load_config()
    rng = random.Random(cfg["seed"])
    n_tr, n_te = cfg["splits"]["train_per_benchmark"], cfg["splits"]["test_per_benchmark"]
    for name, bcfg in cfg["benchmarks"].items():
        print(f"== {name}")
        train, test = BUILDERS[name](bcfg, rng, n_tr, n_te)
        if train:
            write_jsonl(train, PROCESSED / f"{name}_train.jsonl")
        write_jsonl(test, PROCESSED / f"{name}_test.jsonl")
        print(f"   train={len(train)}  test={len(test)}")


if __name__ == "__main__":
    main()
