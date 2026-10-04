"""Query language parser (README section 8).

Grammar (operators are case-insensitive; extra whitespace tolerated):

    "information retrieval"                  -> PHRASE (consecutive, in order)
    information NEAR/3 retrieval             -> unordered |p1 - p2| <= 3
    information ONEAR/3 retrieval            -> ordered  0 < p2 - p1 <= 3
    WINDOW/10(information retrieval system)  -> all terms within a 10-token window
    information retrieval                    -> Boolean AND (rankable)

Rules:
- Query terms go through the SAME tokenizer as documents (a classic bug
  source if not).  Inside a phrase, "state-of-the-art" becomes 4 tokens.
- NEAR operands must be single terms.  A chain of the SAME operator with the
  SAME k ("a NEAR/3 b NEAR/3 c") is accepted as WINDOW/k(a b c); mixed
  operators or mixed k in a chain is a clear parse error.
- Malformed queries (unbalanced quotes, NEAR/ with no number, negative or
  empty k, WINDOW without parentheses, empty query) raise QueryParseError
  with a message that says how to fix it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tokenizer import TokenizerOptions, tokenize


class QueryParseError(ValueError):
    """Raised for malformed queries; message is user-facing."""


@dataclass
class Query:
    kind: str  # "phrase" | "near" | "window" | "and"
    terms: list[str]
    k: int | None = None
    ordered: bool = False
    raw: str = ""

    def describe(self) -> str:
        if self.kind == "phrase":
            return f'phrase "{ " ".join(self.terms) }"'
        if self.kind == "near":
            op = "ONEAR" if self.ordered else "NEAR"
            return f"{self.terms[0]} {op}/{self.k} {self.terms[1]}"
        if self.kind == "window":
            return f"WINDOW/{self.k}({' '.join(self.terms)})"
        return " AND ".join(self.terms)


_NEAR_RE = re.compile(r"^(onear|near)\s*/\s*(\d+)$", re.IGNORECASE)
_WINDOW_RE = re.compile(
    r"^window\s*/\s*(\d+)\s*\((.*)\)$", re.IGNORECASE | re.DOTALL
)


def _tok(text: str, opts: TokenizerOptions) -> list[str]:
    return tokenize(text, opts)


def parse(query: str, opts: TokenizerOptions | None = None) -> Query:
    opts = opts or TokenizerOptions()
    if query is None or not query.strip():
        raise QueryParseError("Empty query. Type at least one term, e.g.:  information retrieval")

    q = query.strip()

    # ---- phrase -----------------------------------------------------------
    if q.startswith('"'):
        if q.count('"') < 2:
            raise QueryParseError(
                'Unbalanced quote. A phrase needs closing quotes:  "information retrieval"'
            )
        m = re.match(r'^"(.*?)"\s*$', q, re.DOTALL)
        if not m:
            raise QueryParseError(
                'A quoted phrase must be the whole query. Example:  "information retrieval"'
            )
        terms = _tok(m.group(1), opts)
        if not terms:
            raise QueryParseError(
                'The phrase contains no indexable terms (only punctuation/stopwords).'
            )
        return Query(kind="phrase", terms=terms, raw=q)

    if q.count('"') % 2 == 1:
        raise QueryParseError(
            'Unbalanced quote. Put "..." around a phrase:  "information retrieval"'
        )

    # ---- WINDOW/w(...) ------------------------------------------------------
    parts = q.split()
    win_head = parts[0] if parts else ""
    m = _WINDOW_RE.match(q)
    if m:
        w = int(m.group(1))
        inner = m.group(2).strip()
        terms = _tok(inner, opts)
        if not terms:
            raise QueryParseError(
                "WINDOW needs at least one term inside the parentheses:  WINDOW/10(a b c)"
            )
        return Query(kind="window", terms=terms, k=w, raw=q)
    if re.match(r"^window\s*/", win_head, re.IGNORECASE) and not m:
        raise QueryParseError(
            'Bad WINDOW syntax. Use:  WINDOW/10(information retrieval system)'
        )

    # ---- NEAR / ONEAR -------------------------------------------------------
    ops = [p for p in parts if _NEAR_RE.match(p)]
    if ops:
        parsed_ops = []
        for p in ops:
            m = _NEAR_RE.match(p)
            parsed_ops.append((m.group(1).lower(), int(m.group(2))))
        same = all(o == parsed_ops[0] for o in parsed_ops)
        if not same:
            raise QueryParseError(
                "Chained proximity operators must share the same operator and k, "
                'e.g. "a NEAR/3 b NEAR/3 c". Mixed chains are not supported; '
                "use WINDOW/k(a b c) instead."
            )
        if len(parts) != 2 * len(ops) + 1:
            raise QueryParseError(
                'Bad NEAR syntax. Use exactly:  term NEAR/3 term    (operands must be single terms)'
            )
        op_name, k = parsed_ops[0]
        operands = [parts[i] for i in range(0, len(parts), 2)]
        terms: list[str] = []
        for opnd in operands:
            toks = _tok(opnd, opts)
            if len(toks) != 1:
                raise QueryParseError(
                    f'NEAR operand "{opnd}" tokenizes to {len(toks)} terms '
                    f"({toks}). NEAR operands must be single terms; quote phrases "
                    'or use WINDOW/k(...).'
                )
            terms.append(toks[0])
        if len(terms) == 2:
            return Query(
                kind="near", terms=terms, k=k, ordered=(op_name == "onear"), raw=q
            )
        # chain of same operator/k -> n-term window semantics (documented)
        return Query(
            kind="window",
            terms=terms,
            k=k,
            ordered=(op_name == "onear"),
            raw=q,
        )

    for p in parts:
        if re.match(r"^(near|onear)\s*/", p, re.IGNORECASE) and not _NEAR_RE.match(p):
            raise QueryParseError(
                f"Malformed proximity operator {p!r}. Use a slash and a "
                "non-negative number:  NEAR/3  or  ONEAR/3"
            )

    # ---- bare Boolean AND -----------------------------------------------------
    terms = _tok(q, opts)
    if not terms:
        raise QueryParseError(
            "The query has no indexable terms (only punctuation/stopwords were typed?)."
        )
    return Query(kind="and", terms=terms, raw=q)
