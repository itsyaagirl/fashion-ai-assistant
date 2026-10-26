"""Baselines we compare against.

KeywordSearch - classic BM25 keyword search (what a normal shop search box does):
                the query is the garment's text, e.g. "Slim fit blazer | Blazer | Dark Blue".
Popularity    - always recommend the best sellers (no personalisation at all).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

STOP = set("a an and the of with in on for to by from at is are this that it its or as be has have".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in STOP and len(t) > 1]


class KeywordSearch:
    """Okapi BM25, written out in full so the baseline is transparent (k1=1.5, b=0.75)."""

    def __init__(self, catalog: pd.DataFrame, k1: float = 1.5, b: float = 0.75):
        self.ids = catalog["article_id"].astype(int).tolist()
        self.docs = [Counter(tokenize(t)) for t in catalog["text"]]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for d, L in zip(self.docs, self.lens):
            s = 0.0
            for w in q:
                f = d.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out

    def search(self, query: str, k: int = 10) -> list[int]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        return [self.ids[i] for i in order[:k]]


def popularity(catalog: pd.DataFrame, k: int = 10) -> list[int]:
    return catalog.nlargest(k, "total_sales")["article_id"].astype(int).tolist()
