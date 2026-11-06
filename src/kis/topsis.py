"""TOPSIS multi-criteria decision model (proposal section 2.7.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# +1 = higher is better (benefit), -1 = lower is better (cost)
DIRECTIONS = {
    "accuracy": +1,
    "bertscore": +1,
    "judge_score": +1,
    "hallucination_rate": -1,
    "latency_p50_ms": -1,
    "peak_gpu_gb": -1,
    "cost_per_1k_usd": -1,
}

# Example deployment profiles. Weights must sum to 1.
PROFILES = {
    "compliance":  {"accuracy": .30, "hallucination_rate": .30, "judge_score": .10,
                    "bertscore": .05, "latency_p50_ms": .10, "peak_gpu_gb": .05, "cost_per_1k_usd": .10},
    "high_volume": {"accuracy": .15, "hallucination_rate": .15, "judge_score": .05,
                    "bertscore": .05, "latency_p50_ms": .25, "peak_gpu_gb": .10, "cost_per_1k_usd": .25},
}


def topsis(matrix: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """matrix: rows = strategies, columns = criteria. Returns closeness score and rank."""
    cols = [c for c in weights if c in matrix.columns]
    X = matrix[cols].astype(float).values
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1
    V = X / norm * w
    d = np.array([DIRECTIONS[c] for c in cols])
    ideal = np.where(d > 0, V.max(axis=0), V.min(axis=0))
    worst = np.where(d > 0, V.min(axis=0), V.max(axis=0))
    s_plus = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    s_minus = np.sqrt(((V - worst) ** 2).sum(axis=1))
    closeness = s_minus / (s_plus + s_minus + 1e-12)
    out = pd.DataFrame({"closeness": closeness.round(4)}, index=matrix.index)
    out["rank"] = out["closeness"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank")


def sensitivity(matrix: pd.DataFrame, weights: dict[str, float], shift: float = 0.2,
                n: int = 200, seed: int = 0) -> pd.Series:
    """Randomly perturb each weight by +/- shift and count how often each strategy ranks first."""
    rng = np.random.default_rng(seed)
    wins = pd.Series(0, index=matrix.index)
    for _ in range(n):
        w = {k: max(v * (1 + rng.uniform(-shift, shift)), 0) for k, v in weights.items()}
        wins[topsis(matrix, w).index[0]] += 1
    return (wins / n).sort_values(ascending=False)
