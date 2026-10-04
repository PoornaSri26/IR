"""Explainable output (README section 11 sample format; Level 2.4 + 3G).

For each hit show:
- the matched POSITIONS of every query term,
- a snippet of the ORIGINAL text with matched words bracketed,
    ...the field of [information] [retrieval] is...
- and for non-matching documents (why-not), how close the terms got:
    `information` at 5, `retrieval` at 41; min distance 36 > k=3.
"""

from __future__ import annotations

from .index import PositionalIndex
from .proximity import minimal_windows
from .query_parser import Query
from .tokenizer import tokenize, tokenize_with_spans

_SNIPPET_PAD = 60


def _snippet(text: str, token_positions: list[int], opts) -> str:
    """Original text with matched token spans bracketed, elided to a window.

    `token_positions` are 1-BASED token positions (as everywhere else).
    """
    if text is None:
        return "(text not stored for this index)"
    spans = tokenize_with_spans(text, opts)
    match_spans = [spans[i - 1] for i in token_positions if 1 <= i <= len(spans)]
    if not match_spans:
        return text[: 2 * _SNIPPET_PAD] + ("..." if len(text) > 2 * _SNIPPET_PAD else "")
    lo = max(0, match_spans[0][1] - _SNIPPET_PAD)
    hi = min(len(text), match_spans[-1][2] + _SNIPPET_PAD)
    parts = []
    cursor = lo
    for tok, s, e in match_spans:
        parts.append(text[cursor:s])
        parts.append(f"[{text[s:e]}]")
        cursor = e
    parts.append(text[cursor:hi])
    out = "".join(parts)
    return ("..." if lo > 0 else "") + out + ("..." if hi < len(text) else "")


def _token_indexes(index: PositionalIndex, doc_id: int) -> dict[int, str]:
    """Map 1-based token position -> term for a document (rebuilt from text)."""
    toks = tokenize(index.doc_text.get(doc_id, ""), index.tokenizer_opts)
    return {i + 1: t for i, t in enumerate(toks)}


def explain_phrase(index: PositionalIndex, doc_id: int, terms: list[str], start: int) -> str:
    pos_line = " ".join(f"{t}@{start + i}" for i, t in enumerate(terms))
    tok_idx = [start + i for i in range(len(terms))]
    snip = _snippet(index.doc_text.get(doc_id, ""), tok_idx, index.tokenizer_opts)
    return f"{index.name(doc_id)} {pos_line}\n       {snip}"


def explain_near(index: PositionalIndex, doc_id: int, terms: list[str], pairs) -> str:
    t1, t2 = terms[0], terms[-1]
    pair_str = " ".join(f"({a},{b})" for a, b in pairs[:6])
    more = f" (+{len(pairs) - 6} more)" if len(pairs) > 6 else ""
    tok_idx = sorted({p for ab in pairs[:6] for p in ab})
    snip = _snippet(index.doc_text.get(doc_id, ""), tok_idx, index.tokenizer_opts)
    return (
        f"{index.name(doc_id)} {t1}@{[a for a, _ in pairs[:6]]} {t2}@{[b for _, b in pairs[:6]]}"
        f" {pair_str}{more}\n       {snip}"
    )


def explain_window(index: PositionalIndex, doc_id: int, terms: list[str], wins) -> str:
    tmap = _token_indexes(index, doc_id)
    left, right, span = wins[0]
    inside = [tmap[p] for p in sorted(tmap) if left <= p <= right]
    tok_idx = [p for p in sorted(tmap) if left <= p <= right]
    snip = _snippet(index.doc_text.get(doc_id, ""), tok_idx, index.tokenizer_opts)
    return (
        f"{index.name(doc_id)} window [{left}..{right}] span={span} terms={inside}"
        f"\n       {snip}"
    )


def explain_and(index: PositionalIndex, doc_id: int, terms: list[str]) -> str:
    pos_line = " ".join(
        f"{t}@{index.positions(t, doc_id)[:8]}" for t in terms if index.has_term(t)
    )
    first_hit = min(
        (index.positions(t, doc_id)[0] for t in terms if index.positions(t, doc_id)),
        default=1,
    )
    snip = _snippet(index.doc_text.get(doc_id, ""), [first_hit], index.tokenizer_opts)
    return f"{index.name(doc_id)} {pos_line}\n       {snip}"


def explain_hit(index: PositionalIndex, query: Query, doc_id: int, payload) -> str:
    if query.kind == "phrase":
        return explain_phrase(index, doc_id, query.terms, payload[0])
    if query.kind == "near":
        return explain_near(index, doc_id, query.terms, payload)
    if query.kind == "window":
        return explain_window(index, doc_id, query.terms, payload)
    return explain_and(index, doc_id, query.terms)


# --------------------------------------------------------------- why-not 3G


def why_not(index: PositionalIndex, query: Query, doc_name: str) -> str:
    """Explain WHY a document does not match, with positions and distances."""
    doc_id = index.id_of(doc_name)
    if doc_id is None:
        return f"why-not: no document named {doc_name!r}"
    if doc_id not in index.doc_len:
        return f"why-not: {doc_name} is not in this index"

    terms = query.terms

    if query.kind == "phrase":
        for i, t in enumerate(terms):
            if not index.tf(t, doc_id):
                return f"why-not: {doc_name} lacks the term {t!r} entirely"
        # best alignment attempt: closest offset mismatch
        base_positions = index.positions(terms[0], doc_id)
        best = None
        for p in base_positions:
            gaps = []
            ok = True
            for i, t in enumerate(terms[1:], start=1):
                ps = index.positions(t, doc_id)
                j = min(ps, key=lambda q: abs(q - (p + i)))
                gaps.append((t, p + i, j, j - (p + i)))
                if j != p + i:
                    ok = False
            if ok:
                return f"why-not: unexpected - {doc_name} DOES match the phrase"
            worst = max(gaps, key=lambda g: abs(g[3]))
            cand = (abs(worst[3]), p, worst)
            if best is None or cand[0] < best[0]:
                best = cand
        _, p, (t, expected, actual, off) = best
        return (
            f'why-not: {doc_name} does not match "{ " ".join(terms) }": closest '
            f"alignment starts {t1_label(terms[0])}@{p} but {t!r} is at {actual}, "
            f"expected {expected} (off by {off:+d})"
        )

    if query.kind == "near":
        t1, t2 = terms[0], terms[-1]
        if not index.tf(t1, doc_id):
            return f"why-not: {doc_name} lacks {t1!r} entirely"
        if not index.tf(t2, doc_id):
            return f"why-not: {doc_name} lacks {t2!r} entirely"
        from .proximity import min_distance

        res = min_distance(index, t1, t2, doc_id, ordered=query.ordered)
        if res is None:
            return f"why-not: {doc_name} has both terms but no valid pair"
        dist, a, b = res
        if dist <= query.k:
            return f"why-not: unexpected - {doc_name} DOES match {query.describe()}"
        return (
            f"why-not: {doc_name} does not match {query.describe()}: "
            f"{t1!r} at {a}, {t2!r} at {b}; min distance = {dist} > k={query.k}"
        )

    if query.kind == "window":
        pos_by = {t: index.positions(t, doc_id) for t in set(terms)}
        missing = [t for t, ps in pos_by.items() if not ps]
        if missing:
            return f"why-not: {doc_name} lacks: {', '.join(repr(m) for m in missing)}"
        need = {}
        for t in terms:
            need[t] = need.get(t, 0) + 1
        wins = minimal_windows(pos_by, 10**9, need)  # minimal cover regardless of w
        if not wins:
            return f"why-not: {doc_name} has all terms but no covering window"
        _, _, span = min(wins, key=lambda x: x[2])
        return (
            f"why-not: {doc_name} does not match {query.describe()}: "
            f"minimal window covering all terms has span {span} > w={query.k}"
        )

    # Boolean AND
    missing = [t for t in set(terms) if not index.tf(t, doc_id)]
    if missing:
        return f"why-not: {doc_name} does not match the AND query; missing: {', '.join(repr(m) for m in missing)}"
    return f"why-not: unexpected - {doc_name} DOES match {query.describe()}"


def t1_label(t: str) -> str:
    return f"`{t}`"
