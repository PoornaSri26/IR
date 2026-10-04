"""Evaluation metric tests — every value hand-checkable."""

import math

from pir.evaluate import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    set_precision,
    set_recall,
)


def test_precision_recall_hand_checked():
    ranked = [1, 2, 3, 4, 5]
    relevant = {2, 4, 6}
    assert precision_at_k(ranked, relevant, 5) == 2 / 5
    assert precision_at_k(ranked, relevant, 1) == 0.0
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 5) == 2 / 3
    assert recall_at_k(ranked, relevant, 100) == 2 / 3
    assert r_precision(ranked, relevant) == precision_at_k(ranked, relevant, 3)


def test_average_precision_hand_checked():
    # hits at ranks 2 and 4: AP = (1/2 + 2/4) / 3 = 1/3
    ranked = [1, 2, 3, 4, 5]
    relevant = {2, 4, 6}
    assert abs(average_precision(ranked, relevant) - 1 / 3) < 1e-12
    assert average_precision([], relevant) == 0.0
    assert average_precision(ranked, set()) == 0.0
    # perfect ranking
    assert average_precision([2, 4, 6, 1, 3], relevant) == 1.0


def test_ndcg_graded_hand_checked():
    grades = {1: 3, 2: 2, 4: 1}
    ranked = [1, 2, 3, 4, 5]
    dcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(5)
    idcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
    assert abs(ndcg_at_k(ranked, grades, 10) - dcg / idcg) < 1e-12
    # perfect order gives 1.0
    assert abs(ndcg_at_k([1, 2, 4], grades, 3) - 1.0) < 1e-12
    assert ndcg_at_k(ranked, {}, 10) == 0.0


def test_set_recall_for_boolean_systems():
    assert set_recall({1, 2, 3}, {2, 3, 4}) == 2 / 3
    assert set_recall(set(), {2}) == 0.0
    assert set_recall({1}, set()) == 0.0


def test_set_precision_for_boolean_systems():
    assert set_precision({1, 2, 3, 4}, {2, 3}) == 0.5
    assert set_precision(set(), {2}) == 0.0
    assert set_precision({2, 3}, {2, 3}) == 1.0
    assert set_precision({9}, {2}) == 0.0
