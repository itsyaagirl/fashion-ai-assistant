"""Standard ranking metrics.

hit@k   - did ANY bought-together accessory appear in the top k? (1 or 0)
mrr     - 1 / position of the first correct item (1st = 1.0, 2nd = 0.5, ...)
ndcg@k  - rewards correct items more the higher they are ranked, 0..1
"""
from __future__ import annotations

import math


def hit_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return float(any(r in relevant for r in ranked[:k]))


def mrr(ranked: list[int], relevant: set[int], k: int | None = None) -> float:
    for i, r in enumerate(ranked[:k] if k else ranked, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], 1) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
