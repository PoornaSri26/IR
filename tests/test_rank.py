"""Ranking tests: BM25 sanity + proximity boost behavior."""

from pir.index import build_index
from pir.rank import BM25, proximity_bonus, rank_bm25, rank_bm25_proximity

DOCS = {
    "a": "apple banana apple cherry",
    "b": "banana banana banana banana",
    "c": "apple apple apple apple apple apple apple apple",
    "d": "cherry durian",
}


def _idx():
    return build_index(DOCS)


def test_bm25_orders_by_query_terms():
    idx = _idx()
    ranked = rank_bm25(idx, [idx.id_of(n) for n in ("a", "b", "c", "d")], ["apple"])
    names = [idx.doc_names[d] for d, _ in ranked]
    # 'c' has 8 apples, 'a' has 2 -> c first; 'b','d' have none -> score 0
    assert names[0] == "c"
    assert names[1] == "a"
    assert ranked[2][1] == 0.0 and ranked[3][1] == 0.0


def test_bm25_idf_familiar():
    idx = _idx()
    scorer = BM25(idx)
    # apple and cherry both have df=2 in this collection
    import math

    expected = math.log(1 + (4 - 2 + 0.5) / (2 + 0.5))
    assert abs(scorer._idf("apple") - expected) < 1e-12
    # a term with lower df gets higher idf
    assert scorer._idf("durian") > scorer._idf("apple")  # durian df=1


def test_bm25_zero_for_missing_terms():
    idx = _idx()
    assert BM25(idx).score(idx.id_of("a"), ["durian"]) == 0.0


def test_proximity_bonus_prefers_close_terms():
    idx = build_index(
        {
            "near": "information retrieval is fun information retrieval again",
            "far": "information x y z retrieval is fun",
        }
    )
    ids = [idx.id_of("near"), idx.id_of("far")]
    bonus_near = proximity_bonus(idx, ids[0], ["information", "retrieval"])
    bonus_far = proximity_bonus(idx, ids[1], ["information", "retrieval"])
    # 'far' has distance 4 > K=3, so its bonus is exactly zero
    assert bonus_near > 0
    assert bonus_far == 0


def test_bm25prox_lambda_zero_equals_bm25():
    idx = _idx()
    ids = list(range(1, 5))
    plain = rank_bm25(idx, ids, ["apple", "banana"])
    prox0 = rank_bm25_proximity(idx, ids, ["apple", "banana"], lam=0.0)
    assert [d for d, _ in plain] == [d for d, _ in prox0]
    assert abs(plain[0][1] - prox0[0][1]) < 1e-12


def test_ranking_ties_broken_by_doc_id():
    idx = build_index({"x": "same words", "y": "same words"})
    ranked = rank_bm25(idx, [idx.id_of("y"), idx.id_of("x")], ["same", "words"])
    assert [idx.doc_names[d] for d, _ in ranked] == ["x", "y"]
