"""Explain module tests: snippets, positions, why-not diagnostics."""

from pir.explain import explain_hit, why_not
from pir.query_parser import parse


def test_snippet_highlights_and_positions(toy_index):
    q = parse('"information retrieval"')
    from pir.phrase import phrase_query

    res = phrase_query(toy_index, q.terms)
    text = explain_hit(toy_index, q, toy_index.id_of("d1"), res[toy_index.id_of("d1")])
    assert "information@1 retrieval@2" in text
    assert "[Information] [retrieval]" in text


def test_why_not_missing_term(toy_index):
    q = parse('"information retrieval"')
    msg = why_not(toy_index, q, "d9")  # d9 has information but no retrieval
    assert "lacks" in msg and "retrieval" in msg


def test_why_not_distance(toy_index):
    q = parse("heat NEAR/2 temperature")
    msg = why_not(toy_index, q, "d9")
    # heat@1, temperature@6 -> min distance 5 > 2
    assert "min distance = 5 > k=2" in msg
    assert "heat" in msg and "temperature" in msg


def test_why_not_unknown_doc(toy_index):
    q = parse("heat transfer")
    assert "no document" in why_not(toy_index, q, "nope")


def test_why_not_phrase_closest_alignment(toy_index):
    q = parse('"boundary layer"')
    msg = why_not(toy_index, q, "d6")  # lacks both terms
    assert "lacks" in msg


def test_explain_near_and_window(toy_index):
    q = parse("shock NEAR/3 wave")
    from pir.proximity import near_query

    res = near_query(toy_index, "shock", "wave", 3, False)
    text = explain_hit(toy_index, q, toy_index.id_of("d7"), res[toy_index.id_of("d7")])
    assert "[shock] [wave]" in text

    q2 = parse("WINDOW/4(x ray machine)")
    from pir.proximity import window_query

    res2 = window_query(toy_index, q2.terms, 4)
    text2 = explain_hit(toy_index, q2, toy_index.id_of("d8"), res2[toy_index.id_of("d8")])
    assert "span=2" in text2 and "[x]-[ray] [machine]" in text2
