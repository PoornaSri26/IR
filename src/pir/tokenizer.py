"""Tokenizer: the single tokenization pipeline used for BOTH documents and queries.

Design decisions (README section 4):
- Case folding: ON by default (all text lowercased before token matching).
- Tokens are maximal runs of [a-z0-9] after lowercasing.
    => punctuation splits tokens ("don't" -> "don", "t"; "x-ray" -> "x", "ray")
    => digits are kept by default ("3.5" -> "3", "5")
- Positions are 1-based: the first token of a document is at position 1.
  (Chosen and documented; only consistency matters, the base is arbitrary.)
- Stopwords are KEPT in the index by default: removing them shifts positions
  and breaks phrases ("king of spain", "to be or not to be").  Removing them
  (or light stemming) is available ONLY as an ablation option (Level 3H).
- A light "toy" stemmer exists ONLY for the ablation study; the main system
  never stems (pure-IR boundary, README section 1).

`tokenize_with_spans` returns (term, start_char, end_char) triples so the
explain module can highlight matches inside the ORIGINAL text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Lowercase-only pattern: documents/queries are lowercased first when case folding.
TOKEN_RE_LOWER = re.compile(r"[a-z0-9]+")
# Case-sensitive pattern used when case folding is disabled (ablation).
TOKEN_RE_CS = re.compile(r"[A-Za-z0-9]+")

# Small stopword list.  Used only when drop_stopwords=True (ablation) or for
# deriving content-word query bigrams in the evaluation (README section 7.2).
STOPWORDS = frozenset(
    """a an and are as at be been being but by for from has have if in into is
    it its of on or so than that the their there these they this to was were
    what when which who will with would""".split()
)

# Suffixes stripped by the toy stemmer (ablation only, NOT in the main system).
_STEM_SUFFIXES = ("ingly", "edly", "ing", "ed", "ies", "es", "s")


@dataclass(frozen=True)
class TokenizerOptions:
    """Configuration shared by documents and queries (recorded in saved indexes)."""

    case_fold: bool = True
    keep_digits: bool = True
    drop_stopwords: bool = False
    stem: bool = False

    def as_dict(self) -> dict:
        return {
            "case_fold": self.case_fold,
            "keep_digits": self.keep_digits,
            "drop_stopwords": self.drop_stopwords,
            "stem": self.stem,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenizerOptions":
        return cls(**{k: d.get(k, cls.__dataclass_fields__[k].default) for k in cls.__dataclass_fields__})


def _stem(token: str) -> str:
    """Very small, clearly-flagged suffix stripper for the ablation study only."""
    if len(token) <= 3:
        return token
    for suf in _STEM_SUFFIXES:
        if token.endswith(suf):
            base = token[: -len(suf)]
            if suf == "ies" and len(base) >= 1:
                return base + "y"
            if suf == "s" and base.endswith("s"):  # keep "class" as "class"
                return token
            if len(base) >= 2:
                return base
    return token


def tokenize(text: str, opts: TokenizerOptions | None = None) -> list[str]:
    """Tokenize `text` into a list of terms (no positions)."""
    opts = opts or TokenizerOptions()
    if text is None:
        return []
    pattern = TOKEN_RE_LOWER if opts.case_fold else TOKEN_RE_CS
    s = text.lower() if opts.case_fold else text
    out: list[str] = []
    for tok in pattern.findall(s):
        if not opts.keep_digits and tok.isdigit():
            continue
        if opts.drop_stopwords and tok in STOPWORDS:
            continue
        if opts.stem:
            tok = _stem(tok)
        out.append(tok)
    return out


def tokenize_with_spans(
    text: str, opts: TokenizerOptions | None = None
) -> list[tuple[str, int, int]]:
    """Tokenize keeping character spans, for snippet highlighting.

    Returns a list of (term, char_start, char_end) with char_end exclusive.
    The terms are exactly `tokenize(text, opts)[i]` for index i.
    """
    opts = opts or TokenizerOptions()
    if text is None:
        return []
    pattern = TOKEN_RE_LOWER if opts.case_fold else TOKEN_RE_CS
    s = text.lower() if opts.case_fold else text
    out: list[tuple[str, int, int]] = []
    for m in pattern.finditer(s):
        tok = m.group(0)
        if not opts.keep_digits and tok.isdigit():
            continue
        if opts.drop_stopwords and tok in STOPWORDS:
            continue
        if opts.stem:
            tok = _stem(tok)
        out.append((tok, m.start(), m.end()))
    return out


def is_stopword(term: str) -> bool:
    return term in STOPWORDS
