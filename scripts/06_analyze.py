"""Analysis layer: summary tables, paired significance tests, and TOPSIS.

Works with whatever is in results/raw/. With only the baseline done it prints the
baseline summary; once 2+ strategies exist for a model it runs the paired tests.

Run: python scripts/06_analyze.py --gpu-usd-per-hour 1.2
Output: results/summary.csv, results/tests.json, results/topsis.csv (when possible)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kis.common import RESULTS
from kis.stats import binary_tests, continuous_tests
from kis.topsis import PROFILES, sensitivity, topsis


def load_results() -> tuple[pd.DataFrame, dict]:
    files = sorted((RESULTS / "raw").glob("*.csv"))
    if not files:
        raise SystemExit("No results yet. Run scripts/05_run_eval.py first.")
    df = pd.concat([pd.read_csv(f, dtype={"tag": "string", "pred_choice": "string"}) for f in files],
                   ignore_index=True)
    df["tag"] = df["tag"].fillna("").astype(str)
    df["condition"] = df["strategy"] + df["tag"].map(lambda t: f"[{t}]" if t else "")
    metas = {}
    for f in files:
        m = Path(str(f).replace(".csv", ".meta.json"))
        if m.exists():
            metas[f.stem] = json.loads(m.read_text())
    return df, metas


def summary(df: pd.DataFrame, metas: dict, gpu_rate: float | None) -> pd.DataFrame:
    g = df.groupby(["model", "condition", "dataset"])
    s = g.agg(n=("item_id", "count"), accuracy=("correct", "mean"), contains=("contains", "mean"),
              hallucination_rate=("hallucinated", "mean"), latency_p50_ms=("latency_ms", "median"),
              latency_p95_ms=("latency_ms", lambda x: x.quantile(.95)),
              latency_p99_ms=("latency_ms", lambda x: x.quantile(.99)),
              mean_latency_ms=("latency_ms", "mean")).reset_index()
    peak = {(m["model"], m["strategy"] + (f"[{m['tag']}]" if m["tag"] else "")): m["peak_gpu_gb"]
            for m in metas.values()}
    s["peak_gpu_gb"] = [peak.get((r.model, r.condition)) for r in s.itertuples()]
    if gpu_rate:
        # cost per 1,000 queries = hourly GPU price x hours needed for 1,000 queries
        s["cost_per_1k_usd"] = gpu_rate * s["mean_latency_ms"] * 1000 / 3.6e6
    return s.round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-usd-per-hour", type=float, default=None)
    args = ap.parse_args()

    df, metas = load_results()
    s = summary(df, metas, args.gpu_usd_per_hour)
    s.to_csv(RESULTS / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(s.to_string(index=False))

    tests = []
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        if sub["condition"].nunique() < 2:
            continue
        sub = sub.rename(columns={"strategy": "_s"}).rename(columns={"condition": "strategy"})
        tests.append({"model": model, "dataset": dataset, **binary_tests(sub, "correct")})
        if sub["hallucinated"].notna().any():
            tests.append({"model": model, "dataset": dataset, **binary_tests(sub, "hallucinated")})
        tests.append({"model": model, "dataset": dataset, **continuous_tests(sub, "latency_ms")})
    if tests:
        (RESULTS / "tests.json").write_text(json.dumps(tests, indent=2))
        print(f"\nSignificance tests written to results/tests.json ({len(tests)} families)")
    else:
        print("\nOnly one condition per model so far - paired tests run once a second strategy exists.")

    # TOPSIS per model, averaged over datasets
    agg = s.groupby(["model", "condition"]).agg(
        accuracy=("accuracy", "mean"), hallucination_rate=("hallucination_rate", "mean"),
        latency_p50_ms=("latency_p50_ms", "mean"), peak_gpu_gb=("peak_gpu_gb", "mean"),
        **({"cost_per_1k_usd": ("cost_per_1k_usd", "mean")} if "cost_per_1k_usd" in s else {}))
    rows = []
    for model, m in agg.groupby(level=0):
        m = m.droplevel(0).dropna(axis=1)
        if len(m) < 2:
            continue
        for profile, w in PROFILES.items():
            r = topsis(m, w)
            stab = sensitivity(m, w)
            for cond, row in r.iterrows():
                rows.append({"model": model, "profile": profile, "condition": cond, **row.to_dict(),
                             "share_ranked_first_under_perturbation": float(stab.get(cond, 0))})
    if rows:
        pd.DataFrame(rows).to_csv(RESULTS / "topsis.csv", index=False)
        print("TOPSIS rankings written to results/topsis.csv")


if __name__ == "__main__":
    main()
