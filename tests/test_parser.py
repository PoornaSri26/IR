"""Query parser tests: grammar, case-insensitivity, clear errors."""

import pytest

from pir.query_parser import QueryParseError, parse


def test_phrase():
    q = parse('"information retrieval"')
    assert q.kind == "phrase" and q.terms == ["information", "retrieval"]


def test_phrase_tokenizes_like_documents():
    # hyphens/digits inside a phrase go through the SAME tokenizer
    q = parse('"X-Ray 3.5"')
    assert q.terms == ["x", "ray", "3", "5"]


def test_near_ordered_unordered_case_insensitive():
    q = parse("shock NEAR/3 wave")
    assert (q.kind, q.k, q.ordered) == ("near", 3, False)
    q = parse("shock onear/3 wave")
    assert (q.kind, q.k, q.ordered) == ("near", 3, True)
    q = parse("shock   Near/3    wave")
    assert q.terms == ["shock", "wave"]


def test_near_operand_tokenized():
    q = parse("Shock NEAR/3 Wave")
    assert q.terms == ["shock", "wave"]  # operands go through the document tokenizer


def test_window():
    q = parse("WINDOW/10(information retrieval system)")
    assert (q.kind, q.k, q.terms) == ("window", 10, ["information", "retrieval", "system"])
    q = parse("window/3(a b)")
    assert q.kind == "window"


def test_near_chain_becomes_window():
    q = parse("a NEAR/4 b NEAR/4 c")
    assert q.kind == "window" and q.terms == ["a", "b", "c"] and q.k == 4


def test_bare_terms_are_and():
    q = parse("information retrieval")
    assert q.kind == "and" and q.terms == ["information", "retrieval"]


def test_bare_word_near_is_a_term():
    q = parse("sitting near me")
    assert q.kind == "and" and q.terms == ["sitting", "near", "me"]


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        '"unbalanced',
        '"a b" extra',
        '""',
        "a NEAR/ b",
        "a NEAR/-1 b",
        "a NEAR/x b",
        "a NEAR/3 b NEAR/5 c",  # mixed k in a chain
        "a NEAR/3 b ONEAR/3 c",  # mixed operators in a chain
        "WINDOW/10 a b",
        'a "b c',  # unbalanced quote mid-query
    ],
)
def test_errors(bad):
    with pytest.raises(QueryParseError):
        parse(bad)


def test_error_messages_are_actionable():
    with pytest.raises(QueryParseError) as e:
        parse("a NEAR/ b")
    assert "NEAR/3" in str(e.value)
    with pytest.raises(QueryParseError) as e:
        parse('"unbalanced')
    assert "quotes" in str(e.value).lower()


def test_multi_token_near_operand_rejected():
    with pytest.raises(QueryParseError):
        parse('"a b" NEAR/3 c')
    with pytest.raises(QueryParseError):
        parse("Shock-Wave NEAR/3 wave")  # hyphen splits into 2 tokens
