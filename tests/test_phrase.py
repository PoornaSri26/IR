"""Phrase query tests (Level 2.2 edge cases)."""

from pir.phrase import phrase_query


def names(index, result):
    return {index.doc_names[d]: ps for d, ps in result.items()}


def test_single_term_phrase_is_all_docs_with_term(toy_index):
    res = names(toy_index, phrase_query(toy_index, ["information"]))
    assert set(res) == {"d1", "d9", "d10"}
    assert res["d1"] == [1, 8, 14]


def test_basic_phrase(toy_index):
    res = names(toy_index, phrase_query(toy_index, ["information", "retrieval"]))
    assert set(res) == {"d1", "d10"}
    assert res["d1"] == [1, 14]  # "Information retrieval" appears twice
    assert res["d10"] == [5]


def test_phrase_not_across_document_boundary(toy_index):
    # d9 ends "...across information" and d10 starts "Retrieval systems..."
    # a positional index can NEVER match across the boundary:
    res = names(toy_index, phrase_query(toy_index, ["across", "information", "retrieval"]))
    assert res == {}


def test_repeated_terms(toy_index):
    assert names(toy_index, phrase_query(toy_index, ["new", "new", "york"])) == {"d5": [1]}
    assert names(toy_index, phrase_query(toy_index, ["buffalo", "buffalo"])) == {"d6": [12, 13]}
    # "new york" matches three times in d5
    assert names(toy_index, phrase_query(toy_index, ["new", "york"])) == {"d5": [2, 8, 10]}


def test_stopword_phrase_works(toy_index):
    assert names(toy_index, phrase_query(toy_index, ["king", "of", "spain"])) == {"d3": [2, 9]}
    assert names(toy_index, phrase_query(toy_index, ["to", "be", "or", "not", "to", "be"])) == {
        "d4": [1]
    }


def test_missing_term_and_empty(toy_index):
    assert phrase_query(toy_index, ["zzz"]) == {}
    assert phrase_query(toy_index, ["information", "zzz"]) == {}
    assert phrase_query(toy_index, []) == {}


def test_case_and_punctuation_insensitive(toy_index):
    # "X-Ray" and "3.5" are handled by the shared tokenizer
    assert names(toy_index, phrase_query(toy_index, ["x", "ray"])) == {"d8": [2]}
    assert names(toy_index, phrase_query(toy_index, ["3", "5"])) == {"d8": [8]}


def test_oracle_on_toy(toy_tokens, toy_index):
    from conftest import bf_phrase

    for terms in (
        ["information", "retrieval"],
        ["new", "new", "york"],
        ["king", "of", "spain"],
        ["data", "data"],
        ["to", "be", "or", "not", "to", "be"],
        ["does", "not", "exist"],
    ):
        expected = {n: ps for n, ps in bf_phrase(toy_tokens, terms).items()}
        got = {toy_index.doc_names[d]: ps for d, ps in phrase_query(toy_index, terms).items()}
        assert got == expected, terms
