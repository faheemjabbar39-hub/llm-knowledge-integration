"""Statistical analysis plan (proposal section 2.6.9).

Binary per-item outcomes (correct, hallucinated): Cochran's Q omnibus -> pairwise McNemar -> Holm.
Continuous per-item outcomes (latency, BERTScore, judge score): Friedman -> pairwise Wilcoxon -> Holm.
All tests are PAIRED: the same items are answered under every strategy.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05


def _paired_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows = items, columns = strategies; keeps only items scored under every strategy."""
    wide = df.pivot_table(index="item_id", columns="strategy", values=metric, aggfunc="first")
    return wide.dropna()


def binary_tests(df: pd.DataFrame, metric: str = "correct") -> dict:
    wide = _paired_table(df, metric).astype(int)
    strategies = list(wide.columns)
    out = {"metric": metric, "n_items": len(wide),
           "proportions": wide.mean().round(4).to_dict()}
    if len(strategies) < 2 or len(wide) == 0:
        out["note"] = "need at least two strategies with shared items"
        return out

    if len(strategies) > 2:
        q = cochrans_q(wide.values)
        out["omnibus"] = {"test": "Cochran's Q", "stat": float(q.statistic),
                          "df": len(strategies) - 1, "p": float(q.pvalue)}

    rows = []
    for a, b in combinations(strategies, 2):
        both = wide[[a, b]]
        n10 = int(((both[a] == 1) & (both[b] == 0)).sum())   # a right, b wrong
        n01 = int(((both[a] == 0) & (both[b] == 1)).sum())   # a wrong, b right
        table = [[int(((both[a] == 1) & (both[b] == 1)).sum()), n10],
                 [n01, int(((both[a] == 0) & (both[b] == 0)).sum())]]
        p = float(mcnemar(table, exact=(n10 + n01) < 25).pvalue) if (n10 + n01) else 1.0
        rows.append({"a": a, "b": b, "diff_pp": round(100 * (both[a].mean() - both[b].mean()), 2),
                     "discordant": n10 + n01, "a_only": n10, "b_only": n01, "p_raw": p})
    out["pairwise"] = _holm(rows)
    return out


def continuous_tests(df: pd.DataFrame, metric: str = "latency_ms") -> dict:
    wide = _paired_table(df, metric).astype(float)
    strategies = list(wide.columns)
    out = {"metric": metric, "n_items": len(wide),
           "median": wide.median().round(3).to_dict(),
           "p95": wide.quantile(0.95).round(3).to_dict(),
           "p99": wide.quantile(0.99).round(3).to_dict()}
    if len(strategies) < 2 or len(wide) == 0:
        out["note"] = "need at least two strategies with shared items"
        return out

    if len(strategies) > 2:
        f = friedmanchisquare(*[wide[s].values for s in strategies])
        out["omnibus"] = {"test": "Friedman", "stat": float(f.statistic),
                          "df": len(strategies) - 1, "p": float(f.pvalue)}
    rows = []
    for a, b in combinations(strategies, 2):
        d = wide[a] - wide[b]
        p = float(wilcoxon(wide[a], wide[b]).pvalue) if np.any(d != 0) else 1.0
        rows.append({"a": a, "b": b, "median_diff": round(float(d.median()), 3), "p_raw": p})
    out["pairwise"] = _holm(rows)
    return out


def _holm(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    reject, p_adj, _, _ = multipletests([r["p_raw"] for r in rows], alpha=ALPHA, method="holm")
    for r, rej, pa in zip(rows, reject, p_adj):
        r["p_holm"] = float(pa)
        r["significant"] = bool(rej)
    return rows
