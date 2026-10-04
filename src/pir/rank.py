"""Ranking: BM25 (rank.py baseline) + optional proximity boost (Level 3C).

BM25 (standard formulation):
    score(q, d) = sum_t  idf(t) * tf(t,d) * (k1 + 1)
                        / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))     (Lucene-style, >= 0)
    k1 = 1.2, b = 0.75 (defaults; documented in README)

Proximity boost (Level 3C, after Metzler & Croft's sequential-dependence
spirit / Tao & Zhai 2007, but deliberately simple and explainable):
    prox(d) = sum over ADJACENT query term pairs (t_i, t_{i+1}) of
              idf_mean(t_i, t_{i+1}) * count of occurrences within distance <= K
    final   = BM25 + lambda * prox(d)
K = 3 by default (adjacent words = distance 1, so a phrase is the strongest
case and NEAR/3 adds a little slack).  lambda = 0 reproduces plain BM25;
lambda is swept in the experiments and reported honestly.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

from .index import PositionalIndex


@dataclass
class BM25:
    index: PositionalIndex
    k1: float = 1.2
    b: float = 0.75
    # Cached at construction time — avoids recomputing sum(doc_len) and
    # log(...) on every score() call across all ranked documents.
    _avgdl: float = field(default=0.0, init=False, repr=False)
    _idf_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._avgdl = self.index.avg_doc_len() or 1.0

    def _idf(self, term: str) -> float:
        v = self._idf_cache.get(term)
        if v is None:
            df = self.index.df(term)
            N = len(self.index.doc_len)
            v = math.log(1.0 + (N - df + 0.5) / (df + 0.5)) if df and N else 0.0
            self._idf_cache[term] = v
        return v

    def score(self, doc_id: int, terms: list[str]) -> float:
        dl = self.index.doc_len.get(doc_id, 0)
        s = 0.0
        for t in terms:
            tf = self.index.tf(t, doc_id)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
            s += self._idf(t) * tf * (self.k1 + 1.0) / denom
        return s

    def rank(self, doc_ids, terms: list[str]) -> list[tuple[int, float]]:
        """Return [(doc_id, score)] sorted by score desc, doc_id asc on ties."""
        scored = [(d, self.score(d, terms)) for d in doc_ids]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored


def count_near_pairs(P1: list[int], P2: list[int], k: int) -> int:
    """Number of unordered pairs with |p1 - p2| <= k, p1 != p2 (O(tf1 log tf2))."""
    n = 0
    for a in P1:
        l = bisect_left(P2, a - k)
        r = bisect_right(P2, a + k)
        for b in P2[l:r]:
            if b != a:
                n += 1
    return n


def proximity_bonus(
    index: PositionalIndex,
    doc_id: int,
    terms: list[str],
    K: int = 3,
) -> float:
    """Pairwise proximity count, weighted by mean idf of each pair."""
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for i in range(len(terms) - 1):
        a, b = terms[i], terms[i + 1]
        key = (a, b)
        if a == b or key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    if not pairs:
        return 0.0
    bonus = 0.0
    for a, b in pairs:
        if not index.has_term(a) or not index.has_term(b):
            continue
        P1, P2 = index.positions(a, doc_id), index.positions(b, doc_id)
        if not P1 or not P2:
            continue
        cnt = count_near_pairs(P1, P2, K)
        if cnt:
            idf = 0.5 * (index.df(a) + index.df(b))
            bonus += cnt / (1.0 + math.log(idf + 1.0))
    return bonus


def rank_bm25(index: PositionalIndex, doc_ids, terms: list[str]) -> list[tuple[int, float]]:
    return BM25(index).rank(doc_ids, terms)


def rank_bm25_proximity(
    index: PositionalIndex,
    doc_ids,
    terms: list[str],
    lam: float = 0.25,
    K: int = 3,
) -> list[tuple[int, float]]:
    scorer = BM25(index)
    scored = [
        (d, scorer.score(d, terms) + lam * proximity_bonus(index, d, terms, K))
        for d in doc_ids
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored
