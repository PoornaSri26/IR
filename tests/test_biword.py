"""Biword index tests + the false-positive behavior that motivates
positional indexes (Level 3E)."""

from pir.biword import build_biword_index, biword_phrase_query
from pir.index import build_index
from pir.phrase import phrase_query


def test_biword_two_word_matches_positional(toy_index, toy_docs):
    bw = build_biword_index(toy_docs)
    for terms in [["information", "retrieval"], ["new", "york"], ["buffalo", "buffalo"]]:
        got = {toy_index.doc_names[d]: ps for d, ps in biword_phrase_query(bw, terms).items()}
        want = {toy_index.doc_names[d]: ps for d, ps in phrase_query(toy_index, terms).items()}
        assert got == want, terms


def test_biword_longer_phrase_with_positional_filter(toy_index, toy_docs):
    bw = build_biword_index(toy_docs)
    terms = ["information", "retrieval", "systems"]
    got = biword_phrase_query(bw, terms, positional=toy_index)
    got_names = {toy_index.doc_names[d] for d, ps in got.items() if ps}
    want_names = {toy_index.doc_names[d] for d in phrase_query(toy_index, terms)}
    assert got_names == want_names


def test_biword_false_positives_without_filter():
    # doc contains "a b" at position 1 and "b c" at positions 4-5, but the
    # phrase "a b c" never occurs; doc-level bigram intersection is a
    # FALSE POSITIVE that only the positional check removes.
    docs = {"x1": "a b z b c", "x2": "a b c q r"}
    bw = build_biword_index(docs)
    positional = build_index(docs)

    def names(result, require_pos=False):
        return {
            positional.doc_names[d]
            for d, ps in result.items()
            if not require_pos or ps
        }

    unfiltered = names(biword_phrase_query(bw, ["a", "b", "c"]))
    filtered = names(
        biword_phrase_query(bw, ["a", "b", "c"], positional=positional),
        require_pos=True,
    )
    truth = names(phrase_query(positional, ["a", "b", "c"]))

    assert "x1" in unfiltered                      # false positive present
    assert filtered == truth                       # removed by positional check
    assert "x2" in truth


def test_biword_needs_two_terms():
    import pytest

    bw = build_biword_index({"a": "x y"})
    with pytest.raises(ValueError):
        biword_phrase_query(bw, ["x"])
