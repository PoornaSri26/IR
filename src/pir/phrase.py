"""Phrase query executor (README section 6).

Phrase "t1 t2 ... tn" (any n >= 1):
1. If any term is missing from the index -> no matches.
2. Candidate docs: intersect doc-id lists, RAREST term first (query planner
   heuristic), short-circuiting when the intersection becomes empty.
3. Offset trick: for each candidate doc, keep starts p of t1 such that
   t_i occurs at p + (i - 1) for every i, i.e. intersect
       positions(t1)  with  {p - i : p in positions(t_i)}
   This handles repeated terms correctly ("new new york").
4. Positions are per document, so a phrase can NEVER match across a document
   boundary by construction (unit-tested).

Returns {doc_id: sorted start positions}.
"""

from __future__ import annotations

from .index import PositionalIndex, intersect_many


def candidate_docs(index: PositionalIndex, terms: list[str]) -> list[int]:
    """Doc-id lists intersected rarest-term-first, with short-circuit."""
    for t in terms:
        if not index.has_term(t):
            return []
    order = sorted(set(terms), key=index.df)
    return intersect_many([index.docs(t) for t in order])


def phrase_query(index: PositionalIndex, terms: list[str]) -> dict[int, list[int]]:
    """Exact consecutive, in-order phrase match.  n == 1 -> all docs with the term."""
    if not terms:
        return {}
    if any(t not in index.postings for t in terms):
        return {}

    docs = candidate_docs(index, terms)
    if not docs:
        return {}

    if len(terms) == 1:
        t = terms[0]
        return {d: index.positions(t, d) for d in docs}

    results: dict[int, list[int]] = {}
    for d in docs:
        starts = set(index.positions(terms[0], d))
        for i, t in enumerate(terms[1:], start=1):
            starts &= {p - i for p in index.positions(t, d)}  # offset trick
            if not starts:
                break
        if starts:
            results[d] = sorted(starts)
    return results
