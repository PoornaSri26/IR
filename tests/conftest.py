"""Shared fixtures + brute-force oracles used by the differential tests.

The oracles re-derive every answer directly from raw token lists (no index
involved).  test_oracle.py asserts index results == oracle results on
randomized corpora; these functions are also reused by targeted unit tests.
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pir.datasets import load_toy  # noqa: E402
from pir.index import build_index  # noqa: E402
from pir.tokenizer import tokenize  # noqa: E402


@pytest.fixture(scope="session")
def toy_docs():
    return load_toy()


@pytest.fixture(scope="session")
def toy_tokens(toy_docs):
    return {name: tokenize(text) for name, text in toy_docs.items()}


@pytest.fixture(scope="session")
def toy_index(toy_docs):
    return build_index(toy_docs)


def names_of(index, ids):
    return {index.doc_names[i] for i in ids}


# --------------------------------------------------------------- brute force


def bf_positions(docs_tokens: dict[str, list[str]]) -> dict[str, dict[str, list[int]]]:
    """term -> name -> 1-based positions, recomputed from raw tokens."""
    out: dict[str, dict[str, list[int]]] = {}
    for name, toks in docs_tokens.items():
        for i, t in enumerate(toks, start=1):
            out.setdefault(t, {}).setdefault(name, []).append(i)
    return out


def bf_phrase(docs_tokens, terms):
    out = {}
    n = len(terms)
    for name, toks in docs_tokens.items():
        starts = [i + 1 for i in range(len(toks) - n + 1) if toks[i : i + n] == terms]
        if starts:
            out[name] = starts
    return out


def bf_near(docs_tokens, t1, t2, k, ordered):
    out = {}
    for name, toks in docs_tokens.items():
        P1 = [i for i, t in enumerate(toks, 1) if t == t1]
        P2 = [i for i, t in enumerate(toks, 1) if t == t2]
        pairs = set()
        for a in P1:
            for b in P2:
                if ordered:
                    if 0 < b - a <= k:
                        pairs.add((a, b))
                elif a != b and abs(a - b) <= k:
                    pairs.add((a, b))
        if pairs:
            out[name] = pairs
    return out


def bf_window(docs_tokens, terms, w):
    """Minimal window by direct scan; multiplicity-aware."""
    need = Counter(terms)
    out = {}
    for name, toks in docs_tokens.items():
        pos_by: dict[str, list[int]] = {}
        for i, t in enumerate(toks, 1):
            pos_by.setdefault(t, []).append(i)
        best = None
        for i in range(len(toks)):
            for j in range(i, min(len(toks), i + w + 1)):
                span = j - i  # positions i+1..j+1
                if all(
                    sum(1 for p in pos_by.get(t, []) if i + 1 <= p <= j + 1) >= m
                    for t, m in need.items()
                ):
                    if best is None or span < best[2]:
                        best = (i + 1, j + 1, span)
        if best:
            out[name] = [best]
    return out


def bf_and(docs_tokens, terms):
    return {
        name
        for name, toks in docs_tokens.items()
        if all(t in Counter(toks) for t in terms)
    }


# --------------------------------------------------- random corpus generator


def random_corpus(rng: random.Random, max_docs=8, max_len=30, vocab=12):
    words = [f"w{i}" for i in range(vocab)]
    docs = {}
    for d in range(rng.randint(1, max_docs)):
        n = rng.randint(3, max_len)
        docs[f"doc{d}"] = [rng.choice(words) for _ in range(n)]
    return docs
