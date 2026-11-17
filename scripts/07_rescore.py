"""Re-mark saved answers with the current marking rules (no model is run).

Every results CSV keeps the model's full reply in the `prediction` column, so when the
marking code in src/kis/tatqa.py is improved, all runs can be marked again consistently.
Only the score columns change: final_answer, correct, contains, f1, abstained, hallucinated.

Run: python scripts/07_rescore.py
"""
from __future__ import annotations

from glob import glob

import pandas as pd

from kis.common import PROCESSED, RESULTS, read_jsonl
from kis.tatqa import score_tatqa

COLS = ["final_answer", "correct", "contains", "f1", "abstained", "hallucinated"]


def main():
    items = {}
    for split in ("test", "dev"):
        path = PROCESSED / f"tatqa_{split}.jsonl"
        if path.exists():
            items.update({it["id"]: it for it in read_jsonl(path)})
    files = sorted(glob(str(RESULTS / "raw" / "*.csv")) + glob(str(RESULTS / "dev" / "*.csv")))
    if not files:
        raise SystemExit("No results to re-mark.")
    report = []
    for f in files:
        d = pd.read_csv(f, dtype={"tag": "string", "final_answer": "string"}, keep_default_na=False,
                        na_values={"retrieval_hit": [""]})
        before = d[["correct", "contains", "hallucinated", "abstained"]].astype(float).mean() * 100
        scores = [score_tatqa(items[i], str(p)) for i, p in zip(d["item_id"], d["prediction"])]
        for c in COLS:
            d[c] = [s[c] for s in scores]
        after = d[["correct", "contains", "hallucinated", "abstained"]].astype(float).mean() * 100
        d.to_csv(f, index=False, lineterminator="\r\n")   # same line endings as the run script (csv module)
        name = f.split("/")[-1].removesuffix(".csv")
        report.append({"run": name, **{f"{k} before": round(before[k], 1) for k in before.index},
                       **{f"{k} after": round(after[k], 1) for k in after.index}})
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.DataFrame(report).to_string(index=False))


if __name__ == "__main__":
    main()
