"""Positional inverted index.

Structure (README section 5):

    postings:  term -> {doc_id -> [pos1, pos2, ...]}   positions ascending, 1-based
    doc_len:   doc_id -> number of tokens
    doc_names: internal int doc_id -> original dataset name (e.g. "1400")

Internal doc ids are ints, assigned in ascending order of the original doc
name (numeric-aware sort), so builds are deterministic.  Position lists are
kept sorted because we append them in scan order; every consumer can rely on
that invariant (merge algorithms need sorted streams).

Persistence formats:
- JSON  (.json)  human-readable, portable, includes texts
- pickle(.pkl)   fast, exact
- varbyte(.pirz) delta + variable-byte compressed postings (Level 3A);
                 stores postings + doc metadata, but NOT raw texts.

Skip pointers (Level 3B): `skip_intersect` intersects two ascending doc-id
lists using sqrt(n) skip pointers, built lazily and cached per list.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import time
from bisect import bisect_right
from dataclasses import dataclass, field

from .tokenizer import TokenizerOptions, tokenize


def natural_key(name: str):
    """Sort key so 'd10' comes after 'd2' and '2' after '1'."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(name)))


@dataclass
class IndexStats:
    documents: int
    tokens: int
    unique_terms: int
    avg_doc_len: float
    build_seconds: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PositionalIndex:
    postings: dict[str, dict[int, list[int]]] = field(default_factory=dict)
    doc_len: dict[int, int] = field(default_factory=dict)
    doc_names: dict[int, str] = field(default_factory=dict)
    doc_text: dict[int, str] = field(default_factory=dict)
    tokenizer_opts: TokenizerOptions = field(default_factory=TokenizerOptions)
    stats: IndexStats | None = None
    _docid_cache: dict[str, list[int]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ basics
    def has_term(self, term: str) -> bool:
        return term in self.postings

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def docs(self, term: str) -> list[int]:
        """Ascending internal doc ids containing `term` (cached)."""
        if term not in self.postings:
            return []
        cached = self._docid_cache.get(term)
        if cached is None:
            cached = sorted(self.postings[term])
            self._docid_cache[term] = cached
        return cached

    def positions(self, term: str, doc_id: int) -> list[int]:
        return self.postings.get(term, {}).get(doc_id, [])

    def tf(self, term: str, doc_id: int) -> int:
        return len(self.positions(term, doc_id))

    def total_tokens(self) -> int:
        return sum(self.doc_len.values())

    def avg_doc_len(self) -> float:
        if not self.doc_len:
            return 0.0
        return self.total_tokens() / len(self.doc_len)

    def name(self, doc_id: int) -> str:
        return self.doc_names.get(doc_id, str(doc_id))

    def id_of(self, doc_name: str) -> int | None:
        for i, n in self.doc_names.items():
            if n == doc_name:
                return i
        return None

    # -------------------------------------------------------------- save/load
    def to_json_dict(self) -> dict:
        return {
            "format": "pir-json",
            "version": 1,
            "tokenizer": self.tokenizer_opts.as_dict(),
            "stats": self.stats.as_dict() if self.stats else None,
            "doc_names": {str(k): v for k, v in self.doc_names.items()},
            "doc_len": {str(k): v for k, v in self.doc_len.items()},
            "doc_text": {str(k): v for k, v in self.doc_text.items()},
            "postings": {
                term: {str(d): ps for d, ps in sorted(docs.items())}
                for term, docs in sorted(self.postings.items())
            },
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "PositionalIndex":
        idx = cls()
        idx.tokenizer_opts = TokenizerOptions.from_dict(d.get("tokenizer", {}))
        st = d.get("stats")
        idx.stats = IndexStats(**st) if st else None
        idx.doc_names = {int(k): v for k, v in d["doc_names"].items()}
        idx.doc_len = {int(k): v for k, v in d["doc_len"].items()}
        idx.doc_text = {int(k): v for k, v in d.get("doc_text", {}).items()}
        idx.postings = {
            term: {int(doc): list(ps) for doc, ps in docs.items()}
            for term, docs in d["postings"].items()
        }
        return idx

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "PositionalIndex":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json_dict(json.load(f))

    def save_pickle(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_pickle(cls, path: str) -> "PositionalIndex":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} is not a PositionalIndex pickle")
        return obj

    def save_compressed(self, path: str) -> None:
        """Delta + variable-byte postings (Level 3A).  Texts are NOT stored."""
        from .compress import encode_postings_varbyte

        meta = pickle.dumps(
            {
                "doc_names": self.doc_names,
                "doc_len": self.doc_len,
                "tokenizer": self.tokenizer_opts.as_dict(),
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        body = encode_postings_varbyte(self.postings)
        with open(path, "wb") as f:
            f.write(b"PIRZ")
            f.write(len(meta).to_bytes(4, "little"))
            f.write(meta)
            f.write(body)

    @classmethod
    def load_compressed(cls, path: str) -> "PositionalIndex":
        from .compress import decode_postings_varbyte

        with open(path, "rb") as f:
            blob = f.read()
        if blob[:4] != b"PIRZ":
            raise ValueError(f"{path}: bad magic, not a .pirz file")
        meta_len = int.from_bytes(blob[4:8], "little")
        meta = pickle.loads(blob[8 : 8 + meta_len])
        postings = decode_postings_varbyte(blob[8 + meta_len :])
        idx = cls()
        idx.postings = postings
        idx.doc_names = meta["doc_names"]
        idx.doc_len = meta["doc_len"]
        idx.tokenizer_opts = TokenizerOptions.from_dict(meta.get("tokenizer", {}))
        return idx


def build_index(
    docs: dict[str, str],
    tokenizer_opts: TokenizerOptions | None = None,
    keep_text: bool = True,
) -> PositionalIndex:
    """Build a positional inverted index from {doc_name: text}.

    Index build is O(total tokens): one hash-map insert (amortized O(1)) per
    token occurrence.  Doc ids are assigned in natural-sorted name order.
    """
    opts = tokenizer_opts or TokenizerOptions()
    idx = PositionalIndex(tokenizer_opts=opts)
    t0 = time.perf_counter()

    names = sorted(docs, key=natural_key)
    total_tokens = 0
    for doc_id, name in enumerate(names, start=1):
        text = docs[name]
        toks = tokenize(text, opts)
        idx.doc_len[doc_id] = len(toks)
        idx.doc_names[doc_id] = name
        if keep_text:
            idx.doc_text[doc_id] = text
        for pos, term in enumerate(toks, start=1):  # 1-based positions
            idx.postings.setdefault(term, {}).setdefault(doc_id, []).append(pos)
        total_tokens += len(toks)

    idx.stats = IndexStats(
        documents=len(names),
        tokens=total_tokens,
        unique_terms=len(idx.postings),
        avg_doc_len=(total_tokens / len(names)) if names else 0.0,
        build_seconds=time.perf_counter() - t0,
    )
    return idx


def intersect_docids(a: list[int], b: list[int]) -> list[int]:
    """Plain two-pointer intersection of ascending doc-id lists: O(|a|+|b|)."""
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def skip_intersect(a: list[int], b: list[int]) -> list[int]:
    """Intersect with skip pointers (Level 3B) — identical result to intersect_docids.

    `a` is the shorter list and carries skip pointers every sqrt(|a|) entries.
    """
    out: list[int] = []
    i = j = 0
    na, nb = len(a), len(b)
    skip = max(1, int(math.isqrt(na)))
    while i < na and j < nb:
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            # try to skip ahead in `a` while still strictly below b[j];
            # land on the first element >= b[j] (never jump past a match)
            nxt = i + skip
            while nxt < na and a[nxt] < b[j]:
                i = nxt
                nxt = i + skip
            i += 1
        else:
            j += 1
    return out


def intersect_many(lists: list[list[int]], use_skips: bool = False) -> list[int]:
    """Intersect ascending doc-id lists, rarest (shortest) first."""
    if not lists:
        return []
    lists = sorted(lists, key=len)
    cur = lists[0]
    for other in lists[1:]:
        if not cur:
            return []
        cur = skip_intersect(cur, other) if use_skips else intersect_docids(cur, other)
    return cur
