"""Charts of every condition run so far. Rerun after each new model or strategy and the
charts grow automatically (one panel per model, one colour per strategy).

Run:    python scripts/08_plot_results.py      (after scripts/06_analyze.py)
Output: results/figures/*.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kis.common import RESULTS

# Colour follows the STRATEGY everywhere (fixed order, validated for colour-blind safety).
STRATEGY_ORDER = ["base", "qlora", "rag", "cag"]
COLORS = {"base": "#2a78d6", "qlora": "#eb6834", "rag": "#1baf7a", "cag": "#eda100"}
LABELS = {"base": "Base", "qlora": "QLoRA", "rag": "RAG", "cag": "CAG"}
MODEL_ORDER = ["llama8b", "mistral7b", "llama70b"]
MODEL_LABELS = {"llama8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B", "llama70b": "Llama-3.1-70B"}
MODEL_MARKERS = {"llama8b": "o", "mistral7b": "s", "llama70b": "^"}
DATASET_LABELS = {"mmlu_fin": "MMLU finance", "truthfulqa": "TruthfulQA", "nq": "Natural Qs",
                  "hotpotqa": "HotpotQA", "domain": "SEC/ESMA"}

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def _base_strategy(cond: str) -> str:
    return cond.split("[")[0]


def _ordered(values, order):
    return [v for v in order if v in values] + sorted(v for v in values if v not in order)


def _bar_label(ax, bars, fmt):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK2)


def _legend(fig, strategies):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS.get(s, MUTED)) for s in strategies]
    fig.legend(handles, [LABELS.get(s, s) for s in strategies], loc="upper right", frameon=False,
               ncol=len(strategies), title="Strategy", title_fontsize=9)


def grouped_by_dataset(s: pd.DataFrame, metric: str, title: str, ylabel: str, fname: str,
                       datasets=None, pct=True):
    s = s.copy()
    s["strategy"] = s["condition"].map(_base_strategy)
    if datasets:
        s = s[s["dataset"].isin(datasets)]
    s = s.dropna(subset=[metric])
    if s.empty:
        return None
    models = _ordered(s["model"].unique(), MODEL_ORDER)
    strategies = _ordered(s["strategy"].unique(), STRATEGY_ORDER)
    dsets = _ordered(s["dataset"].unique(), list(DATASET_LABELS))
    # one row per model (small multiples stacked), width grows with the number of bars
    n_bars = len(dsets) * len(strategies)
    fig, axes = plt.subplots(len(models), 1, figsize=(max(6.5, 0.42 * n_bars + 2), 0.6 + 3.4 * len(models)),
                             sharey=True, squeeze=False)
    axes = axes.T
    width = 0.8 / len(strategies)
    ymax = s[metric].max() * (100 if pct else 0.001)
    for ax, m in zip(axes[0], models):
        x = np.arange(len(dsets))
        for i, st in enumerate(strategies):
            vals = [s[(s.model == m) & (s.strategy == st) & (s.dataset == d)][metric].mean() for d in dsets]
            vals = np.array(vals, dtype=float) * (100 if pct else 0.001)
            bars = ax.bar(x + (i - (len(strategies) - 1) / 2) * width, vals, width * 0.92,
                          color=COLORS.get(st, MUTED), label=LABELS.get(st, st))
            _bar_label(ax, bars, (lambda v: f"{v:.0f}") if pct else (lambda v: f"{v:.1f}s"))
        ax.set_xticks(x, [DATASET_LABELS.get(d, d) for d in dsets])
        ax.set_title(MODEL_LABELS.get(m, m), loc="left")
        ax.set_ylim(0, ymax * 1.18 if ymax else 1)
        ax.tick_params(axis="x", length=0)
    for ax in axes[0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    _legend(fig, strategies)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    out = RESULTS / "figures" / fname
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def tradeoff(s: pd.DataFrame, fname="accuracy_vs_cost.png"):
    if "cost_per_1k_usd" not in s:
        return None
    s = s.copy()
    s["strategy"] = s["condition"].map(_base_strategy)
    agg = s.groupby(["model", "condition", "strategy"], as_index=False).agg(
        accuracy=("accuracy", "mean"), cost=("cost_per_1k_usd", "mean"))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sparse = len(agg) <= 4
    for r in agg.itertuples():
        ax.scatter(r.cost, r.accuracy * 100, s=90, color=COLORS.get(r.strategy, MUTED),
                   marker=MODEL_MARKERS.get(r.model, "o"), edgecolor=SURFACE, linewidth=2, zorder=3)
        if sparse:  # direct labels only while the chart is sparse; otherwise the legends carry identity
            ax.annotate(f"{MODEL_LABELS.get(r.model, r.model)} · {LABELS.get(r.strategy, r.strategy)}",
                        (r.cost, r.accuracy * 100), xytext=(7, 4), textcoords="offset points",
                        fontsize=8, color=INK2)
    strategies = _ordered(agg["strategy"].unique(), STRATEGY_ORDER)
    models = _ordered(agg["model"].unique(), MODEL_ORDER)
    h1 = [plt.Line2D([], [], ls="", marker="o", ms=8, color=COLORS.get(st, MUTED)) for st in strategies]
    h2 = [plt.Line2D([], [], ls="", marker=MODEL_MARKERS.get(m, "o"), ms=8, color=MUTED) for m in models]
    leg1 = ax.legend(h1, [LABELS.get(st, st) for st in strategies], title="Strategy (colour)",
                     loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(h2, [MODEL_LABELS.get(m, m) for m in models], title="Model (shape)",
              loc="upper left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9, title_fontsize=9)
    ax.set_xlabel("Cost per 1,000 questions (US$)")
    ax.set_ylabel("Mean accuracy across test sets (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlim(0, agg["cost"].max() * 1.6)
    ax.set_ylim(0, max(agg["accuracy"].max() * 100 * 1.25, 10))
    ax.set_title("Accuracy vs cost (top-left is best)", loc="left")
    fig.tight_layout()
    out = RESULTS / "figures" / fname
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    path = RESULTS / "summary.csv"
    if not path.exists():
        raise SystemExit("results/summary.csv not found - run scripts/06_analyze.py first.")
    s = pd.read_csv(path)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    made = [
        grouped_by_dataset(s, "accuracy", "Accuracy (exact match) by test set",
                           "Correct answers (%)", "accuracy.png"),
        grouped_by_dataset(s, "hallucination_rate", "Hallucination rate on TruthfulQA (lower is better)",
                           "Answers repeating a falsehood (%)", "hallucination.png", datasets=["truthfulqa"]),
        grouped_by_dataset(s, "latency_p50_ms", "Typical response time (median, lower is better)",
                           "Seconds per answer", "latency.png", pct=False),
        tradeoff(s),
    ]
    for m in made:
        if m:
            print(f"saved {m.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
