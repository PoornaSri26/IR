"""Proximity and window query executors (README section 7).

Distance convention (the #1 viva confusion, so it is stated everywhere):

    ADJACENT WORDS HAVE DISTANCE 1.  Token positions are 1-based and the
    distance between two token occurrences is |p1 - p2|.  Therefore:

    - unordered  NEAR/k   : p1 != p2 and |p1 - p2| <= k
    - ordered    ONEAR/k  : 0 < p2 - p1 <= k      (t2 occurs AFTER t1)
    - phrase     "a b"    == a ONEAR/1 b          (exactly)

    Some systems (Google-style proximity, Lucene "slop") count INTERVENING
    words instead; there, adjacent words have distance 0.  With that
    convention NEAR/0 would equal the phrase.  Our convention keeps
    NEAR/1 <-> phrase symmetry clean and is stated in the README.

Notes:
- p1 != p2 matters only when both operands are the SAME term ("data NEAR/2
  data"): a position can never be paired with itself.
- k = 0 is therefore impossible for any pair (distinct positions differ by
  >= 1) and yields an empty result; k larger than any document behaves like
  a Boolean AND.  Both are unit-tested.

Two-pointer merges are O(|P1| + |P2|) per document for MATCH DETECTION
(`near_unordered_match` / Manning-style); full PAIR ENUMERATION uses
bisect windows (O(tf1 * log tf2)) so no valid pair is ever missed.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from .index import PositionalIndex
from .phrase import candidate_docs


# ------------------------------------------------------------------- pairs


def near_pairs(
    P1: list[int], P2: list[int], k: int, ordered: bool = False
) -> list[tuple[int, int]]:
    """All (p1, p2) pairs satisfying the proximity predicate.

    Unordered: p1 != p2 and |p1 - p2| <= k.
    Ordered:   0 < p2 - p1 <= k.
    P1 and P2 must be ascending position lists.
    """
    if k < 0:
        return []
    pairs: list[tuple[int, int]] = []
    for a in P1:
        if ordered:
            lo, hi = a + 1, a + k
        else:
            lo, hi = a - k, a + k
        l = bisect_left(P2, lo)
        r = bisect_right(P2, hi)
        for b in P2[l:r]:
            if b == a:  # same occurrence (same term); never self-pair
                continue
            pairs.append((a, b))
    return pairs


# --------------------------------------------------- two-pointer detection


def near_unordered_match(P1: list[int], P2: list[int], k: int) -> bool:
    """Two-pointer merge: O(|P1| + |P2|), match DETECTION only.

    Same-term operands (P1 == P2 lists) put the same position in both streams.
    A pair at equal positions is invalid (p1 != p2), so before advancing we
    check whether that shared position still has a valid partner later in
    EITHER stream; if it does not, both pointers can safely move on.
    """
    i = j = 0
    while i < len(P1) and j < len(P2):
        d = P2[j] - P1[i]
        if d == 0:  # same position: the equal pair is not valid
            v = P1[i]
            if j + 1 < len(P2) and P2[j + 1] - v <= k:
                return True
            if i + 1 < len(P1) and P1[i + 1] - v <= k:
                return True
            i += 1
            j += 1  # v can pair with nothing: drop it from both streams
        elif abs(d) <= k:
            return True
        elif P1[i] < P2[j]:
            i += 1
        else:
            j += 1
    return False


def near_ordered_match(P1: list[int], P2: list[int], k: int) -> bool:
    """Two-pointer detection for 0 < p2 - p1 <= k: O(|P1| + |P2|).

    The merge must advance the SMALLER of the two positions: p2 that is
    already left of p1 can never pair with p1 or any later p1, and p1 that is
    more than k left of p2 can never pair with it or any earlier p2.
    """
    i = j = 0
    while i < len(P1) and j < len(P2):
        a, b = P1[i], P2[j]
        if a == b:  # equal positions: not a valid ordered pair
            # only the SECOND stream can pair forward (p2 > p1 is required);
            # a later p1 at this same value would need b > p1, which ordering forbids
            if j + 1 < len(P2) and P2[j + 1] - a <= k:
                return True
            i += 1
            j += 1  # a can pair with nothing: drop it from both streams
        elif 0 < b - a <= k:
            return True
        elif a < b:  # b - a > k: p1 is too early, move to the next p1
            i += 1
        else:  # a > b: p2 is already passed, move to the next p2
            j += 1
    return False


# ------------------------------------------------------------------ queries


def near_query(
    index: PositionalIndex,
    t1: str,
    t2: str,
    k: int,
    ordered: bool = False,
) -> dict[int, list[tuple[int, int]]]:
    """NEAR/k over the whole index -> {doc_id: [(p1, p2), ...]}."""
    if k < 0 or not index.has_term(t1) or not index.has_term(t2):
        return {}
    out: dict[int, list[tuple[int, int]]] = {}
    for d in candidate_docs(index, [t1, t2]):
        pairs = near_pairs(index.positions(t1, d), index.positions(t2, d), k, ordered)
        if pairs:
            out[d] = pairs
    return out


def min_distance(
    index: PositionalIndex, t1: str, t2: str, doc_id: int, ordered: bool = False
) -> tuple[int, int, int] | None:
    """Smallest distance between the two terms in one doc.

    Unordered: min |p1 - p2| over p1 != p2;  ordered: min (p2 - p1) > 0.
    Returns (distance, p1, p2) or None if one term is absent / no valid pair.
    Used by the why-not explainer (Level 3G).
    """
    P1, P2 = index.positions(t1, doc_id), index.positions(t2, doc_id)
    if not P1 or not P2:
        return None
    best: tuple[int, int, int] | None = None
    if ordered:
        for a in P1:
            j = bisect_right(P2, a)  # first p2 strictly greater than a
            if j < len(P2):
                cand = (P2[j] - a, a, P2[j])
                if best is None or cand[0] < best[0]:
                    best = cand
    else:
        for a in P1:
            j = bisect_left(P2, a)
            for b in P2[max(0, j - 1) : j + 1]:
                if b == a:
                    continue
                cand = (abs(b - a), a, b)
                if best is None or cand[0] < best[0]:
                    best = cand
    return best


# ----------------------------------------------------------- window queries


def minimal_windows(
    positions_by_term: dict[str, list[int]],
    w: int,
    need: dict[str, int] | None = None,
) -> list[tuple[int, int, int]]:
    """Sliding-window minimal covers over merged position streams.

    Finds windows [left_pos, right_pos] with right_pos - left_pos <= w that
    contain at least one occurrence of EVERY query term; a term appearing m
    times in the query (WINDOW/w(data data)) needs m DISTINCT occurrences in
    the window.

    Returns a list of (left_pos, right_pos, span) for windows that are
    minimal (shrinking by one token breaks coverage).  Complexity is linear
    in the number of events (total positions of query terms in the doc).
    """
    if need is None:
        need = {t: 1 for t in positions_by_term}
    if w < 0 or not need:
        return []
    if any(not positions_by_term.get(t) for t in need):
        return []

    events: list[tuple[int, str]] = sorted(
        (p, t) for t, ps in positions_by_term.items() for p in ps
    )

    have: dict[str, int] = {}
    covered_terms = 0
    left = 0
    windows: list[tuple[int, int, int]] = []
    for right, (rp, rt) in enumerate(events):
        have[rt] = have.get(rt, 0) + 1
        if have[rt] == need[rt]:
            covered_terms += 1
        while covered_terms == len(need):
            span = rp - events[left][0]
            if span <= w:
                windows.append((events[left][0], rp, span))
            lt = events[left][1]
            have[lt] -= 1
            if have[lt] < need[lt]:
                covered_terms -= 1
            left += 1
    return windows


def window_query(
    index: PositionalIndex, terms: list[str], w: int
) -> dict[int, list[tuple[int, int, int]]]:
    """WINDOW/w(t1 ... tn) -> {doc_id: [(left, right, span), ...]}.

    Generalizes pairwise proximity to n terms (Level 3D).  For two terms and
    unordered semantics, WINDOW/k(a b) is exactly `a NEAR/k b`.
    """
    if w < 0 or not terms:
        return {}
    if any(t not in index.postings for t in set(terms)):
        return {}
    need: dict[str, int] = {}
    for t in terms:
        need[t] = need.get(t, 0) + 1
    out: dict[int, list[tuple[int, int, int]]] = {}
    for d in candidate_docs(index, terms):
        pos_by = {t: index.positions(t, d) for t in set(terms)}
        wins = minimal_windows(pos_by, w, need)
        if wins:
            out[d] = wins
    return out
