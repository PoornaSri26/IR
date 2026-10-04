"""Evaluation metrics implemented from scratch (~README section 7.1).

- precision_at_k, recall_at_k, R-precision
- average_precision -> MAP (mean over queries)
- nDCG@k with graded relevance (gain 2^rel - 1, log2 discount)

All functions take `ranked` = ordered list of doc ids and `relevant` =
a set (binary) or dict {doc: grade} (graded).  A tiny hand-checkable example
is unit-tested in tests/test_evaluate.py and cross-checked against
pytrec_eval conventions.
"""

from __future__ import annotations

import math


def _rel(grade) -> float:
    """Grade to binary relevance; a set membership is binary 1/0."""
    if isinstance(grade, bool):
        return 1.0 if grade else 0.0
    return 1.0 if grade > 0 else 0.0


def precision_at_k(ranked: list, relevant, k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for d in top if _is_rel(d, relevant))
    return hits / k


def recall_at_k(ranked: list, relevant, k: int) -> float:
    total = len(relevant)
    if total == 0:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for d in top if _is_rel(d, relevant))
    return hits / total


def r_precision(ranked: list, relevant) -> float:
    return precision_at_k(ranked, relevant, len(relevant))


def _is_rel(doc, relevant) -> bool:
    if isinstance(relevant, dict):
        return relevant.get(doc, 0) > 0
    return doc in relevant


def average_precision(ranked: list, relevant) -> float:
    """AP = (1/|relevant|) * sum over relevant hits of precision@rank."""
    total = len(relevant)
    if total == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, d in enumerate(ranked, start=1):
        if _is_rel(d, relevant):
            hits += 1
            s += hits / i
    return s / total


def map_score(ranked_lists: list[list], relevants: list) -> float:
    """Mean Average Precision over queries (relevance = binary)."""
    if not ranked_lists:
        return 0.0
    return sum(average_precision(r, rel) for r, rel in zip(ranked_lists, relevants)) / len(
        ranked_lists
    )


def ndcg_at_k(ranked: list, grades: dict, k: int) -> float:
    """nDCG@k for graded relevance: gain 2^rel - 1, discount 1/log2(rank+1).

    `grades` maps doc_id -> graded relevance where HIGHER is MORE relevant.
    (Cranfield's raw scale runs the other way; datasets.py converts it and
    documents the conversion.)
    """
    if k <= 0 or not grades:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        g = grades.get(d, 0)
        dcg += (2.0**g - 1.0) / math.log2(i + 1)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2.0**g - 1.0) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def set_recall(matched_docs, relevant) -> float:
    """Recall of a Boolean result SET: |matched ∩ relevant| / |relevant|."""
    if not relevant:
        return 0.0
    return len(set(matched_docs) & set(relevant)) / len(relevant)


def set_precision(matched_docs, relevant) -> float:
    """Precision of a Boolean result SET: |matched ∩ relevant| / |matched|.

    This is the metric where the classic proximity tradeoff is visible: as k
    shrinks, the returned set gets more precise and less complete.
    """
    matched = set(matched_docs)
    if not matched:
        return 0.0
    return len(matched & set(relevant)) / len(matched)
