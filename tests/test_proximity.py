"""Proximity query tests: ordered/unordered, edge k values, metamorphic laws."""

from pir.phrase import phrase_query
from pir.proximity import (
    near_ordered_match,
    near_pairs,
    near_query,
    near_unordered_match,
)

from conftest import names_of


def test_unordered_basic(toy_index):
    res = near_query(toy_index, "shock", "wave", 3, ordered=False)
    assert names_of(toy_index, res) == {"d7"}
    assert set(next(iter(res.values()))) == {(2, 3), (13, 14)}


def test_same_term_never_self_pairs(toy_index):
    # data at 1,2,5,8 in d6
    res = near_query(toy_index, "data", "data", 2, ordered=False)
    assert names_of(toy_index, res) == {"d6"}
    pairs = set(next(iter(res.values())))
    assert pairs == {(1, 2), (2, 1)}
    assert (1, 1) not in pairs and (2, 2) not in pairs


def test_k_zero_impossible(toy_index):
    # distinct positions differ by >= 1, so NEAR/0 can never match
    assert near_query(toy_index, "shock", "wave", 0, ordered=False) == {}
    assert near_query(toy_index, "data", "data", 0, ordered=False) == {}


def test_k_larger_than_doc_behaves_like_AND(toy_index):
    res = near_query(toy_index, "shock", "wave", 10**9, ordered=False)
    assert names_of(toy_index, res) == {"d7"}
    assert len(next(iter(res.values()))) == 4  # all 2x2 pairs, no self-pairs possible


def test_ordered_vs_unordered(toy_index):
    unordered = near_query(toy_index, "information", "retrieval", 2, ordered=False)
    ordered = near_query(toy_index, "information", "retrieval", 2, ordered=True)
    # unordered is a superset of ordered
    assert names_of(toy_index, ordered) <= names_of(toy_index, unordered)
    # d9: information@9 alone -> no retrieval nearby either way
    assert "d9" not in names_of(toy_index, unordered)


def test_onear1_equals_phrase(toy_index):
    for t1, t2 in [("information", "retrieval"), ("new", "york"), ("buffalo", "buffalo")]:
        ph = phrase_query(toy_index, [t1, t2])
        on = near_query(toy_index, t1, t2, 1, ordered=True)
        assert names_of(toy_index, ph) == names_of(toy_index, on), (t1, t2)


def test_monotone_in_k(toy_index):
    prev = set(near_query(toy_index, "data", "warehouse", 1, False))
    for k in (2, 3, 5, 8):
        cur = set(near_query(toy_index, "data", "warehouse", k, False))
        assert prev <= cur, f"NEAR/{k - 1} results must be a subset of NEAR/{k}"
        prev = cur


def test_two_pointer_agrees_with_pair_enumeration():
    import random

    rng = random.Random(7)
    for _ in range(500):
        P1 = sorted(rng.sample(range(1, 40), rng.randint(0, 6)))
        P2 = sorted(rng.sample(range(1, 40), rng.randint(0, 6)))
        k = rng.randint(0, 8)
        by_pairs = bool(near_pairs(P1, P2, k, ordered=False))
        assert near_unordered_match(P1, P2, k) == by_pairs
        by_pairs_o = bool(near_pairs(P1, P2, k, ordered=True))
        assert near_ordered_match(P1, P2, k) == by_pairs_o


def test_pairs_match_oracle(toy_tokens, toy_index):
    from conftest import bf_near

    for t1, t2, k, ordered in [
        ("shock", "wave", 3, False),
        ("shock", "wave", 3, True),
        ("data", "data", 2, False),
        ("information", "retrieval", 2, False),
        ("heat", "temperature", 4, False),
    ]:
        got = {
            toy_index.doc_names[d]: set(ps)
            for d, ps in near_query(toy_index, t1, t2, k, ordered).items()
        }
        want = bf_near(toy_tokens, t1, t2, k, ordered)
        assert got == want, (t1, t2, k, ordered)
