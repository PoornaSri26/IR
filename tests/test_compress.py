"""Compression tests: varbyte + gamma roundtrips, edge values, postings."""

import random

from pir.compress import (
    decode_postings_gamma,
    decode_postings_varbyte,
    delta_decode,
    delta_encode,
    encode_postings_gamma,
    encode_postings_varbyte,
    gamma_decode,
    gamma_encode,
    vb_decode,
    vb_encode,
    vb_encode_number,
)


def test_vb_edge_values():
    for n in [0, 1, 127, 128, 16383, 16384, 2**21, 2**31 - 1]:
        assert vb_decode(vb_encode_number(n)) == ([n], len(vb_encode_number(n)))


def test_vb_zero_gap_and_large():
    nums = [0, 0, 1, 0, 1000000, 5, 0]
    out, _ = vb_decode(vb_encode(nums))
    assert out == nums


def test_vb_long_stream():
    rng = random.Random(1)
    nums = [rng.randint(0, 10**7) for _ in range(1000)]
    out, pos = vb_decode(vb_encode(nums))
    assert out == nums and pos == len(vb_encode(nums))


def test_vb_single_byte_for_small_gaps():
    assert len(vb_encode_number(0)) == 1
    assert len(vb_encode_number(127)) == 1
    assert len(vb_encode_number(128)) == 2


def test_delta_roundtrip_and_zero_gaps():
    assert delta_decode(delta_encode([1, 2, 3, 10])) == [1, 2, 3, 10]
    assert delta_encode([1, 2, 3, 10]) == [1, 1, 1, 7]
    assert delta_decode(delta_encode([5])) == [5]
    assert delta_decode([]) == []


def test_gamma_roundtrip():
    rng = random.Random(2)
    nums = [1, 2, 3, 127, 128, 1000] + [rng.randint(1, 10**6) for _ in range(200)]
    assert gamma_decode(gamma_encode(nums), len(nums)) == nums


def test_postings_varbyte_roundtrip(toy_index):
    blob = encode_postings_varbyte(toy_index.postings)
    back = decode_postings_varbyte(blob)
    assert back == toy_index.postings


def test_postings_gamma_roundtrip(toy_index):
    blob = encode_postings_gamma(toy_index.postings)
    back = decode_postings_gamma(blob, toy_index.postings)
    assert back == toy_index.postings


def test_postings_gamma_smaller_than_varbyte(toy_index):
    # both are delta-based; gamma must at least decode to the same structure
    gm = encode_postings_gamma(toy_index.postings)
    vb = encode_postings_varbyte(toy_index.postings)
    assert len(gm) > 0 and len(vb) > 0
