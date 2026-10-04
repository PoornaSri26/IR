"""Differential (oracle) tests + metamorphic properties — the strongest
accuracy claim: for randomized corpora and queries, index results are equal
to brute-force scans over raw token lists (1,000+ randomized checks).
"""

import random

from pir.index import build_index
from pir.phrase import phrase_query
from pir.proximity import near_query, window_query
from pir.tokenizer import tokenize

from conftest import bf_and, bf_near, bf_phrase, bf_window, random_corpus

N_CORPORA = 200
QUERIES_PER = 6  # -> 1,200 randomized query checks


def _rand_terms(rng, vocab):
    words = [f"w{i}" for i in range(vocab)]
    n = rng.choice([1, 2, 2, 3, 3, 4])
    terms = [rng.choice(words) for _ in range(n)]
    if rng.random() < 0.25 and n >= 2:
        terms[1] = terms[0]  # force repeated term
    return terms


def test_oracle_differential():
    rng = random.Random(2026)
    checks = 0
    for _ in range(N_CORPORA):
        docs = random_corpus(rng)
        toks = {name: ts for name, ts in docs.items()}
        index = build_index({name: " ".join(ts) for name, ts in docs.items()})
        for _ in range(QUERIES_PER):
            terms = _rand_terms(rng, 12)
            k = rng.randint(0, 7)
            w = rng.randint(0, 8)

            got_phrase = {
                index.doc_names[d]: ps for d, ps in phrase_query(index, terms).items()
            }
            assert got_phrase == bf_phrase(toks, terms), terms

            got_near = {
                index.doc_names[d]: set(ps)
                for d, ps in near_query(index, terms[0], terms[-1], k, False).items()
            }
            assert got_near == bf_near(toks, terms[0], terms[-1], k, False), (terms, k)

            got_ordered = {
                index.doc_names[d]: set(ps)
                for d, ps in near_query(index, terms[0], terms[-1], k, True).items()
            }
            assert got_ordered == bf_near(
                toks, terms[0], terms[-1], k, True
            ), (terms, k)

            got_win = {index.doc_names[d] for d in window_query(index, terms, w)}
            want_win = set(bf_window(toks, terms, w))
            assert got_win == want_win, (terms, w)
            checks += 4
    assert checks == N_CORPORA * QUERIES_PER * 4


def test_metamorphic_properties():
    """Laws that must hold for every corpus/query (section 8.3 of the spec)."""
    rng = random.Random(99)
    for _ in range(100):
        docs = random_corpus(rng)
        toks = {n: ts for n, ts in docs.items()}
        index = build_index({n: " ".join(ts) for n, ts in docs.items()})
        terms = _rand_terms(rng, 12)
        if len(terms) < 2:
            terms = terms + terms  # proximity laws need two operands

        def names(result):
            return {index.doc_names[d] for d in result}

        # 1. phrase results are a subset of ONEAR/1 results...
        ph = names(phrase_query(index, terms)) if len(terms) >= 2 else set()
        on1 = names(near_query(index, terms[0], terms[1], 1, True))
        if len(terms) == 2:
            # ...and ONEAR/1 EQUALS the phrase for 2 terms
            assert ph == on1, terms
        else:
            assert ph <= on1 | set(), terms

        # 2. NEAR/k subset of NEAR/(k+1)
        nk = names(near_query(index, terms[0], terms[1], 3, False))
        nk1 = names(near_query(index, terms[0], terms[1], 4, False))
        assert nk <= nk1

        # 3. ordered subset of unordered (same k)
        ok = names(near_query(index, terms[0], terms[1], 3, True))
        assert ok <= nk

        # 4. every phrase hit is in the AND of ALL query terms, and every
        # proximity hit is in the AND of the two operands actually used
        and_docs = bf_and(toks, terms)
        and_pair = bf_and(toks, terms[:2])
        for hit in ph:
            assert hit in and_docs
        for hit in nk:
            assert hit in and_pair


def test_tokens_of_index_match_raw_tokenization(toy_docs, toy_index):
    """The index must be built from exactly the shared tokenizer's output."""
    for name, text in toy_docs.items():
        doc_id = toy_index.id_of(name)
        toks = tokenize(text)
        assert toy_index.doc_len[doc_id] == len(toks)
        rebuilt = [None] * len(toks)
        for t in set(toks):
            for p in toy_index.positions(t, doc_id):
                rebuilt[p - 1] = t  # 1-based positions
        assert rebuilt == toks
