"""Step 5: summary tables, paired significance tests and the TOPSIS decision model.

Works with whatever is in results/raw/: with one strategy it prints the summary only;
once a model has two or more strategies it also runs the paired tests and TOPSIS.

Run:    python scripts/05_analyze.py
Output: results/summary.csv          one row per model and strategy
        results/summary_by_type.csv  the same, split by kind of answer and by where the answer is
        results/tests.json           Cochran's Q / McNemar and Friedman / Wilcoxon, Holm-corrected
        results/topsis.csv           strategy ranking per model for two deployment profiles
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from kis.common import RESULTS, load_config
from kis.stats import binary_tests, continuous_tests
from kis.topsis import PROFILES, sensitivity, topsis


def _condition(strategy: str, tag) -> str:
    return strategy + (f"[{tag}]" if isinstance(tag, str) and tag else "")


def load_results() -> tuple[pd.DataFrame, dict]:
    files = sorted((RESULTS / "raw").glob("*.csv"))
    if not files:
        raise SystemExit("No results yet. Run scripts/03_run_eval.py first.")
    df = pd.concat([pd.read_csv(f, dtype={"tag": "string"}) for f in files], ignore_index=True)
    df["tag"] = df["tag"].fillna("").astype(str)
    df["condition"] = [_condition(s, t) for s, t in zip(df["strategy"], df["tag"])]
    metas = {}
    for f in files:
        m = f.with_suffix(".meta.json")
        if m.exists():
            meta = json.loads(m.read_text())
            metas[(meta["model"], _condition(meta["strategy"], meta.get("tag", "")))] = meta
    return df, metas


def _setup_seconds(meta: dict) -> float | None:
    total = 0.0
    for v in (meta.get("setup") or {}).values():
        total += sum(v.values()) if isinstance(v, dict) else (v or 0)
    return round(total, 1) if total else None


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return df.groupby(by).agg(
        n=("item_id", "count"), accuracy=("correct", "mean"), accuracy_lenient=("contains", "mean"),
        f1=("f1", "mean"), hallucination_rate=("hallucinated", "mean"), abstain_rate=("abstained", "mean"),
        retrieval_hit=("retrieval_hit", "mean"), latency_p50_ms=("latency_ms", "median"),
        latency_p95_ms=("latency_ms", lambda x: x.quantile(.95)),
        latency_p99_ms=("latency_ms", lambda x: x.quantile(.99)),
        mean_latency_ms=("latency_ms", "mean"), mean_output_tokens=("output_tokens", "mean")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-usd-per-hour", type=float, default=None, help="default: value in the config")
    args = ap.parse_args()
    rate = args.gpu_usd_per_hour or load_config().get("gpu_usd_per_hour")

    df, metas = load_results()
    s = summarise(df, ["model", "condition"])
    s["peak_gpu_gb"] = [metas.get((r.model, r.condition), {}).get("peak_gpu_gb") for r in s.itertuples()]
    s["setup_s"] = [_setup_seconds(metas.get((r.model, r.condition), {})) for r in s.itertuples()]
    if rate:   # cost per 1,000 questions = hourly GPU price x hours needed to answer 1,000 questions
        s["cost_per_1k_usd"] = rate * s["mean_latency_ms"] * 1000 / 3.6e6
    s = s.round(4)
    s.to_csv(RESULTS / "summary.csv", index=False)
    show = ["model", "condition", "n", "accuracy", "accuracy_lenient", "hallucination_rate", "abstain_rate",
            "retrieval_hit", "latency_p50_ms", "peak_gpu_gb"] + (["cost_per_1k_usd"] if rate else [])
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(s[show].to_string(index=False))

    parts = []
    for col in ("answer_type", "answer_from"):
        t = summarise(df, ["model", "condition", col]).rename(columns={col: "group"})
        t.insert(2, "breakdown", col)
        parts.append(t)
    pd.concat(parts, ignore_index=True).round(4).to_csv(RESULTS / "summary_by_type.csv", index=False)

    # paired tests and TOPSIS use the four main conditions only (no extra tagged runs)
    main_df = df[df["tag"] == ""]
    tests = []
    for model, sub in main_df.groupby("model"):
        if sub["strategy"].nunique() < 2:
            continue
        for metric in ("correct", "hallucinated"):
            tests.append({"model": model, **binary_tests(sub, metric)})
        for metric in ("latency_ms", "f1"):
            tests.append({"model": model, **continuous_tests(sub, metric)})
    if tests:
        (RESULTS / "tests.json").write_text(json.dumps(tests, indent=2))
        print(f"\nSignificance tests written to results/tests.json ({len(tests)} test families)")
        for t in tests:
            if t["metric"] == "correct" and "pairwise" in t:
                print(f"\n{t['model']} accuracy, {t['n_items']} paired questions: " +
                      (f"{t['omnibus']['test']} p = {t['omnibus']['p']:.4f}" if "omnibus" in t else ""))
                for r in t["pairwise"]:
                    print(f"   {r['a']:>6} vs {r['b']:<6} {r['diff_pp']:+6.1f} points   Holm p = {r['p_holm']:.4f}"
                          f"   {'real difference' if r['significant'] else 'not significant'}")
    else:
        print("\nOnly one strategy per model so far: paired tests run once a second strategy exists.")

    rows = []
    main_s = s[~s["condition"].str.contains(r"\[")].set_index(["model", "condition"])
    criteria = [c for c in ("accuracy", "f1", "hallucination_rate", "latency_p50_ms", "peak_gpu_gb",
                            "cost_per_1k_usd") if c in main_s.columns]
    for model, m in main_s[criteria].groupby(level=0):
        m = m.droplevel(0).dropna(axis=1)
        if len(m) < 2:
            continue
        for profile, w in PROFILES.items():
            r = topsis(m, w)
            stab = sensitivity(m, w)
            for cond, row in r.iterrows():
                rows.append({"model": model, "profile": profile, "condition": cond,
                             "closeness": row["closeness"], "rank": int(row["rank"]),
                             "share_ranked_first_when_weights_change": float(stab.get(cond, 0))})
    if rows:
        pd.DataFrame(rows).to_csv(RESULTS / "topsis.csv", index=False)
        print("\nTOPSIS rankings written to results/topsis.csv")


if __name__ == "__main__":
    main()
