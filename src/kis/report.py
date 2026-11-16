"""Notebook report for ONE strategy: how good is it, and how does it compare with Base (and RAG)?

Use in a notebook cell:
    from kis.report import show
    show("base")      # or "rag", "cag", "qlora"

It reads whatever has finished in results/raw/, so it can run while some models are not done.
"""
from __future__ import annotations

import json
import os
from glob import glob

import numpy as np
import pandas as pd

from kis.common import RESULTS
from kis.stats import binary_tests

TOTAL = 504
ORDER_S = ["base", "rag", "cag", "qlora"]
ORDER_M = ["llama8b", "mistral7b", "qwen7b"]
NAMES_S = {"base": "Base", "rag": "RAG", "cag": "CAG", "qlora": "QLoRA"}
NAMES_M = {"llama8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B", "qwen7b": "Qwen2.5-7B"}
COLORS = {"base": "#2a78d6", "rag": "#1baf7a", "cag": "#eda100", "qlora": "#eb6834"}
GPU_USD_PER_HOUR = 0.35
IDK = "Said I don't know %"
TYPE_NAMES = {"span": "Value/words from report", "multi-span": "Several values",
              "arithmetic": "Calculation", "count": "Counting"}
SOURCE_NAMES = {"table": "Answer in table", "text": "Answer in text", "table-text": "Needs table + text"}
COMPARE_WITH = {"base": [], "rag": ["base"], "cag": ["base", "rag"], "qlora": ["base", "rag", "cag"]}


def _load() -> tuple[pd.DataFrame, dict]:
    files = sorted(glob(str(RESULTS / "raw" / "*.csv")))
    if not files:
        return pd.DataFrame(), {}
    df = pd.concat([pd.read_csv(f, dtype={"tag": "string"}) for f in files], ignore_index=True)
    df["tag"] = df["tag"].fillna("")
    df = df[df["tag"] == ""]                       # main runs only (extra runs such as --no-docs are left out)
    peak = {}
    for f in files:
        m = f.replace(".csv", ".meta.json")
        if os.path.exists(m):
            meta = json.load(open(m))
            peak[(meta["model"], meta["strategy"])] = meta.get("peak_gpu_gb")
    return df, peak


def _summary(d: pd.DataFrame, peak: dict, strategy: str) -> pd.DataFrame:
    g = d.groupby("model")
    t = pd.DataFrame({
        "Questions": g.size(),
        "Accuracy (strict) %": g["correct"].mean() * 100,
        "Accuracy (lenient) %": g["contains"].mean() * 100,
        "F1 (text answers)": g["f1"].mean(),
        "Hallucination %": g["hallucinated"].mean() * 100,
        IDK: g["abstained"].mean() * 100,
        "Search found right extract %": g["retrieval_hit"].mean() * 100,
        "Median time (s)": g["latency_ms"].median() / 1000,
        "Slow answers p95 (s)": g["latency_ms"].quantile(.95) / 1000,
        "Prompt length (tokens)": g["prompt_tokens"].mean(),
        "Mean time (s)": g["latency_ms"].mean() / 1000,
    })
    t["Peak GPU memory (GB)"] = [peak.get((m, strategy)) for m in t.index]
    t["Cost per 1,000 questions ($)"] = GPU_USD_PER_HOUR * t["Mean time (s)"] * 1000 / 3600
    t = t.drop(columns="Mean time (s)")
    if strategy not in ("rag", "qlora"):
        t = t.drop(columns="Search found right extract %")
    t = t.reindex([m for m in ORDER_M if m in t.index] + [m for m in t.index if m not in ORDER_M])
    return t.rename(index=NAMES_M).rename_axis("Model")


def _fmt(t: pd.DataFrame):
    f = {c: "{:.1f}" for c in t.columns if "%" in c}
    f.update({"F1 (text answers)": "{:.3f}", "Median time (s)": "{:.2f}", "Slow answers p95 (s)": "{:.2f}",
              "Prompt length (tokens)": "{:,.0f}", "Peak GPU memory (GB)": "{:.1f}",
              "Cost per 1,000 questions ($)": "{:.3f}", "Questions": "{:,.0f}"})
    return t.style.format({k: v for k, v in f.items() if k in t.columns}, na_rep="-")


def show(strategy: str) -> pd.DataFrame | None:
    """Print tables, tests, a chart and a plain-language summary for one strategy."""
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    strategy = strategy.lower()
    name = NAMES_S.get(strategy, strategy)
    df, peak = _load()
    d = df[df["strategy"] == strategy] if len(df) else df
    if d.empty:
        print(f"No {name} results yet: run the {name} cell for at least one model first.")
        return None
    models = [m for m in ORDER_M if m in d.model.unique()]

    # 1. progress
    display(Markdown(f"## {name} results\n### 1. Progress: questions answered (out of {TOTAL})"))
    prog = d.groupby("model").size().reindex(models)
    print("   ".join(f"{NAMES_M.get(m, m)}: {n}" for m, n in prog.items()))
    if (prog < TOTAL).any():
        print("Not finished yet: the numbers below are provisional until every model shows 504.")

    # 2. main table
    t = _summary(d, peak, strategy)
    display(Markdown(f"### 2. How good is {name}?"))
    display(_fmt(t))

    # 3. by kind of question
    for col, names, title in [("answer_type", TYPE_NAMES, "kind of answer"),
                              ("answer_from", SOURCE_NAMES, "where the answer is")]:
        p = (d.groupby(["model", col])["correct"].mean() * 100).unstack(col)
        p = p.reindex(index=models, columns=[c for c in names if c in p.columns])
        p = p.rename(index=NAMES_M, columns=names).rename_axis("Model")
        counts = d.drop_duplicates("item_id")[col].map(names).value_counts()
        display(Markdown(f"### 3. Accuracy (strict, %) by {title}"))
        print("Questions per group:", ", ".join(f"{k} {v}" for k, v in counts.items()))
        display(p.style.format("{:.1f}", na_rep="-"))

    # 4. comparison with earlier strategies, on the same questions
    others = [s for s in COMPARE_WITH.get(strategy, []) if s in set(df.strategy)]
    rows = []
    for model in models:
        for other in others:
            pair = df[(df.model == model) & df.strategy.isin([strategy, other])]
            if pair.strategy.nunique() < 2:
                continue
            for metric, label in [("correct", "Accuracy"), ("hallucinated", "Hallucination"),
                                  ("abstained", "Said I don't know")]:
                res = binary_tests(pair, metric)
                if not res.get("pairwise"):
                    continue
                r = res["pairwise"][0]
                diff = r["diff_pp"] if r["a"] == strategy else -r["diff_pp"]
                rows.append({"Model": NAMES_M.get(model, model), "Compared with": NAMES_S[other],
                             "Measure": label, f"{name} %": res["proportions"][strategy] * 100,
                             f"Other %": res["proportions"][other] * 100, "Difference (points)": diff,
                             "Paired questions": res["n_items"], "p (McNemar)": r["p_raw"],
                             "Real difference?": "YES" if r["p_raw"] < 0.05 else "no"})
    if strategy != "base":
        display(Markdown(f"### 4. Is {name} really better? (same questions compared one by one)"))
        if rows:
            cmp = pd.DataFrame(rows).set_index(["Model", "Compared with", "Measure"])
            display(cmp.style.format({f"{name} %": "{:.1f}", "Other %": "{:.1f}", "Difference (points)": "{:+.1f}",
                                      "p (McNemar)": "{:.4f}"}))
            print(f"Difference = {name} minus the other strategy, in percentage points.")
            print("'YES' means p < 0.05: very unlikely to be chance. (The final analysis in Step 6 also applies "
                  "the Holm correction across all comparisons.)")
        else:
            print(f"Run {' / '.join(NAMES_S[s] for s in COMPARE_WITH[strategy])} for the same model to compare.")

    # 5. chart: this strategy next to the ones it is compared with
    shown = [s for s in ORDER_S if s in [strategy] + others]
    sub = df[df.strategy.isin(shown) & df.model.isin(models)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (col, title) in zip(axes, [("correct", "Accuracy (strict), higher is better"),
                                       ("hallucinated", "Hallucination, lower is better"),
                                       ("latency_ms", "Median time per answer (s), lower is better")]):
        x, w = np.arange(len(models)), 0.8 / len(shown)
        for i, s in enumerate(shown):
            q = [sub[(sub.model == m) & (sub.strategy == s)][col] for m in models]
            vals = [v.median() / 1000 if col == "latency_ms" else v.mean() * 100 for v in q]
            bars = ax.bar(x + (i - (len(shown) - 1) / 2) * w, vals, w * 0.92, color=COLORS.get(s, "#898781"),
                          label=NAMES_S.get(s, s), alpha=1.0 if s == strategy else 0.55)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.annotate(f"{v:.1f}" if col == "latency_ms" else f"{v:.0f}",
                                (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                                ha="center", fontsize=8)
        ax.set_xticks(x, [NAMES_M.get(m, m) for m in models])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("seconds" if col == "latency_ms" else "% of questions")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(shown), loc="upper center", bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS / "figures" / f"{strategy}_overview.png", dpi=150)
    plt.show()

    # 6. plain words
    display(Markdown("### 6. Summary in plain words"))
    for model in models:
        r = t.loc[NAMES_M.get(model, model)]
        text = (f"**{NAMES_M.get(model, model)}** with {name}: {r['Accuracy (strict) %']:.1f}% correct, "
                f"{r['Hallucination %']:.1f}% wrong answers, said I don't know {r[IDK]:.1f}%, "
                f"{r['Median time (s)']:.1f}s per answer.")
        if "Search found right extract %" in t.columns and not np.isnan(r["Search found right extract %"]):
            text += f" Search found the right report extract {r['Search found right extract %']:.0f}% of the time."
        for row in rows:
            if row["Model"] == NAMES_M.get(model, model) and row["Measure"] == "Accuracy":
                verdict = "a real difference" if row["Real difference?"] == "YES" else "not a significant difference"
                text += f" Compared with {row['Compared with']}: {row['Difference (points)']:+.1f} points ({verdict})."
        display(Markdown(text))

    t.to_csv(RESULTS / f"summary_{strategy}.csv")
    print(f"\nSaved: results/summary_{strategy}.csv and results/figures/{strategy}_overview.png")
    return t
