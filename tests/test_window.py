"""Window query tests (Level 3D) incl. duplicate-term multiplicity."""

from pir.proximity import minimal_windows, window_query

from conftest import names_of


def test_basic_window(toy_index):
    res = window_query(toy_index, ["x", "ray", "machine"], 4)
    assert names_of(toy_index, res) == {"d8"}
    left, right, span = res[next(iter(res))][0]
    assert (left, right) == (2, 4) and span == 2  # max-min, adjacent = 1


def test_window_equals_near_for_two_terms(toy_index):
    from pir.proximity import near_query

    # documented consistency: WINDOW/k(a b) == a NEAR/k b (unordered)
    for a, b, w in [("shock", "wave", 3), ("data", "warehouse", 5)]:
        wins = {d for d in window_query(toy_index, [a, b], w)}
        nears = {d for d in near_query(toy_index, a, b, w, ordered=False)}
        assert wins == nears, (a, b, w)


def test_window_dup_terms_need_distinct_occurrences(toy_index):
    # WINDOW/1(data data): two DISTINCT occurrences within distance 1
    res = window_query(toy_index, ["data", "data"], 1)
    assert names_of(toy_index, res) == {"d6"}  # data@1,2 adjacent
    res0 = window_query(toy_index, ["data", "data"], 0)
    assert res0 == {}


def test_window_missing_term(toy_index):
    assert window_query(toy_index, ["shock", "zzz"], 5) == {}
    assert window_query(toy_index, [], 5) == {}


def test_minimal_windows_span_semantics():
    # covering all three terms needs b@2 ... c@11 (span 9) or a@1 ... c@11 (span 10)
    pos = {"a": [1, 10], "b": [2], "c": [11]}
    spans = {s for _, _, s in minimal_windows(pos, 10)}
    assert spans == {9, 10}
    assert minimal_windows(pos, 8) == []  # best possible cover is span 9
    # two terms only: a@1,b@2 adjacent -> span 1
    spans2 = {s for _, _, s in minimal_windows({"a": [1, 10], "b": [2]}, 1)}
    assert spans2 == {1}
    assert minimal_windows({"a": [1], "b": [2]}, 0) == []  # adjacent words are distance 1


def test_oracle_on_toy(toy_tokens, toy_index):
    from conftest import bf_window

    for terms, w in [
        (["x", "ray", "machine"], 4),
        (["data", "data"], 1),
        (["information", "retrieval", "needs"], 4),
        (["king", "spain", "madrid"], 6),
    ]:
        want = bf_window(toy_tokens, terms, w)  # {doc name: [(left, right, span)]}
        got = window_query(toy_index, terms, w)
        got_names = {toy_index.doc_names[d] for d in got}
        assert got_names == set(want), (terms, w)
        # minimal span agreement
        for d, wins in got.items():
            name = toy_index.doc_names[d]
            oracle_span = want[name][0][2]
            assert min(s for _, _, s in wins) == oracle_span
