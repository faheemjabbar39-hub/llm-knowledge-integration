"""Charts of every condition run so far. Rerun after each new model or strategy and the
charts grow automatically (one panel per model, one colour per strategy).

Run:    python scripts/06_plot_results.py      (after scripts/05_analyze.py)
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
MODEL_ORDER = ["llama8b", "mistral7b", "qwen7b"]
MODEL_LABELS = {"llama8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B", "qwen7b": "Qwen2.5-7B"}
MODEL_MARKERS = {"llama8b": "o", "mistral7b": "s", "qwen7b": "^"}

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def _ordered(values, order):
    return [v for v in order if v in values] + sorted(v for v in values if v not in order)


def _bar_label(ax, bars, fmt):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK2)


def _legend(fig, strategies):
    """Legend on its own line under the title, so it never collides with the title on narrow charts."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS.get(s, MUTED)) for s in strategies]
    fig.legend(handles, [LABELS.get(s, s) for s in strategies], loc="upper left", frameon=False,
               ncol=len(strategies), bbox_to_anchor=(0.005, 1 - 0.42 / fig.get_figheight()),
               handlelength=1.2, columnspacing=1.4)


def grouped(long: pd.DataFrame, group_order: list[str], title: str, ylabel: str, fname: str, pct=True):
    """Bars per strategy for each group on the x axis; one panel per model.
    `long` has columns model, strategy, group, value (value is a share 0-1, or milliseconds)."""
    long = long.dropna(subset=["value"])
    if long.empty:
        return None
    models = _ordered(long["model"].unique(), MODEL_ORDER)
    strategies = _ordered(long["strategy"].unique(), STRATEGY_ORDER)
    groups = [g for g in group_order if g in set(long["group"])]
    scale = 100 if pct else 0.001
    n_bars = len(groups) * len(strategies)
    fig, axes = plt.subplots(len(models), 1, figsize=(max(6.5, 0.42 * n_bars + 2), 0.6 + 3.4 * len(models)),
                             sharey=True, squeeze=False)
    axes = axes.T
    width = 0.8 / len(strategies)
    ymax = long["value"].max() * scale
    for ax, m in zip(axes[0], models):
        x = np.arange(len(groups))
        for i, st in enumerate(strategies):
            vals = [long[(long.model == m) & (long.strategy == st) & (long.group == g)]["value"].mean()
                    for g in groups]
            vals = np.array(vals, dtype=float) * scale
            bars = ax.bar(x + (i - (len(strategies) - 1) / 2) * width, vals, width * 0.92,
                          color=COLORS.get(st, MUTED), label=LABELS.get(st, st))
            _bar_label(ax, bars, (lambda v: f"{v:.0f}") if pct else (lambda v: f"{v:.1f}s"))
        ax.set_xticks(x, groups)
        ax.set_title(MODEL_LABELS.get(m, m), loc="left")
        ax.set_ylim(0, ymax * 1.18 if ymax else 1)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylabel)
    fig.suptitle(title, x=0.012, y=1 - 0.12 / fig.get_figheight(), ha="left", va="top", fontsize=12,
                 fontweight="bold")
    _legend(fig, strategies)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
    out = RESULTS / "figures" / fname
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _long(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Wide summary -> long form: one row per (model, strategy, group)."""
    parts = [df[["model", "strategy"]].assign(group=label, value=df[col]) for col, label in columns.items()
             if col in df]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["model", "strategy", "group", "value"])


def tradeoff(s: pd.DataFrame, fname="accuracy_vs_cost.png"):
    if "cost_per_1k_usd" not in s:
        return None
    agg = s.rename(columns={"cost_per_1k_usd": "cost"}).dropna(subset=["accuracy", "cost"])
    if agg.empty:
        return None
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
    ax.set_ylabel("Accuracy (%)")
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
        raise SystemExit("results/summary.csv not found - run scripts/05_analyze.py first.")
    s = pd.read_csv(path)
    s = s[~s["condition"].str.contains(r"\[")].copy()      # the four main conditions only
    s["strategy"] = s["condition"]
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(RESULTS / "summary_by_type.csv")
    t = t[(t["breakdown"] == "answer_type") & ~t["condition"].str.contains(r"\[")]
    names = {"span": "Text or value\nfrom the report", "multi-span": "Several values",
             "arithmetic": "Calculation", "count": "Counting"}
    by_type = pd.DataFrame({"model": t["model"], "strategy": t["condition"],
                            "group": t["group"].map(names), "value": t["accuracy"]})
    overall = s[["model", "strategy"]].assign(group="All questions", value=s["accuracy"])

    made = [
        grouped(pd.concat([overall, by_type], ignore_index=True), ["All questions"] + list(names.values()),
                "Accuracy by kind of answer (strict marking)", "Correct answers (%)", "accuracy.png"),
        grouped(_long(s, {"hallucination_rate": "Wrong answer given\n(hallucination)",
                          "abstain_rate": "Said \"I don't know\""}),
                ["Wrong answer given\n(hallucination)", "Said \"I don't know\""],
                "Wrong answers and refusals (lower hallucination is better)", "Share of questions (%)",
                "hallucination.png"),
        grouped(_long(s, {"latency_p50_ms": "Typical answer\n(median)", "latency_p95_ms": "Slow answer\n(95th percentile)"}),
                ["Typical answer\n(median)", "Slow answer\n(95th percentile)"],
                "Response time (lower is better)", "Seconds per answer", "latency.png", pct=False),
        tradeoff(s),
    ]
    for m in made:
        if m:
            print(f"saved {m.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
