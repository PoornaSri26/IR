"""Index build, structure, persistence and skip-pointer tests."""

import random

from pir.index import build_index, intersect_docids, skip_intersect
from pir.tokenizer import TokenizerOptions


def test_stats_toy(toy_index):
    st = toy_index.stats
    assert st.documents == 10
    assert st.tokens == sum(toy_index.doc_len.values())
    assert st.unique_terms == len(toy_index.postings)
    assert abs(st.avg_doc_len - st.tokens / 10) < 1e-9
    assert st.build_seconds >= 0


def test_structure_invariants(toy_index):
    for term, docs in toy_index.postings.items():
        for d, ps in docs.items():
            assert ps == sorted(ps), "positions must be ascending"
            assert len(ps) == len(set(ps)), "positions must be unique"
            assert 1 <= ps[0]  # 1-based
            assert ps[-1] <= toy_index.doc_len[d]
    # deterministic doc ids follow natural name order
    names = [toy_index.doc_names[i] for i in range(1, 11)]
    assert names == [f"d{i}" for i in range(1, 11)]
    assert toy_index.id_of("d10") == 10


def test_df_tf_positions(toy_index):
    assert toy_index.df("information") == 3  # d1 (x3), d9, d10
    assert toy_index.tf("information", 1) == 3
    assert toy_index.positions("information", 1) == [1, 8, 14]
    assert toy_index.positions("missingterm", 1) == []


def test_roundtrip_json_pickle_pirz(toy_index, tmp_path):
    queries = [
        ("phrase", ["information", "retrieval"]),
        ("near", "shock", "wave", 3, False),
        ("near", "data", "data", 2, False),
        ("window", ["x", "ray", "machine"], 4),
    ]
    from pir.phrase import phrase_query
    from pir.proximity import near_query, window_query

    def battery(idx):
        return (
            phrase_query(idx, ["information", "retrieval"]),
            phrase_query(idx, ["new", "new", "york"]),
            near_query(idx, "shock", "wave", 3, False),
            near_query(idx, "data", "data", 2, False),
            window_query(idx, ["x", "ray", "machine"], 4),
        )

    base = battery(toy_index)

    p_json = str(tmp_path / "idx.json")
    p_pkl = str(tmp_path / "idx.pkl")
    p_pirz = str(tmp_path / "idx.pirz")
    toy_index.save_json(p_json)
    toy_index.save_pickle(p_pkl)
    toy_index.save_compressed(p_pirz)

    for loader in (
        lambda: type(toy_index).load_json(p_json),
        lambda: type(toy_index).load_pickle(p_pkl),
        lambda: type(toy_index).load_compressed(p_pirz),
    ):
        loaded = loader()
        assert battery(loaded) == base, "round-trip must preserve all query results"
        assert loaded.stats is None or loaded.stats.documents == toy_index.stats.documents


def test_pirz_smaller_than_json(tmp_path):
    # measured on a larger synthetic collection: the toy corpus is too small
    # for the metadata overhead to be amortized
    import os

    rng = random.Random(3)
    words = [f"w{i}" for i in range(200)]
    docs = {
        f"doc{i}": " ".join(rng.choice(words) for _ in range(200)) for i in range(300)
    }
    idx = build_index(docs)
    pj = str(tmp_path / "big.json")
    pz = str(tmp_path / "big.pirz")
    idx.save_json(pj)
    idx.save_compressed(pz)
    assert os.path.getsize(pz) < os.path.getsize(pj) / 2


def test_load_rejects_bad_magic(tmp_path):
    import pytest

    from pir.index import PositionalIndex

    p = tmp_path / "bad.pirz"
    p.write_bytes(b"NOPE" + b"\x00" * 8)
    with pytest.raises(ValueError):
        PositionalIndex.load_compressed(str(p))


def test_tokenizer_opts_recorded(toy_docs, tmp_path):
    idx = build_index(toy_docs, TokenizerOptions(drop_stopwords=True, stem=True))
    assert idx.tokenizer_opts.drop_stopwords and idx.tokenizer_opts.stem
    p = str(tmp_path / "opt.json")
    idx.save_json(p)
    back = type(idx).load_json(p)
    assert back.tokenizer_opts.drop_stopwords and back.tokenizer_opts.stem
    assert back.postings == idx.postings


def test_skip_intersect_matches_plain():
    rng = random.Random(42)
    for _ in range(300):
        a = sorted(rng.sample(range(1, 500), rng.randint(0, 80)))
        b = sorted(rng.sample(range(1, 500), rng.randint(0, 80)))
        assert skip_intersect(a, b) == intersect_docids(a, b)
    assert skip_intersect([], [1, 2]) == []
    assert skip_intersect([1], []) == []
    assert skip_intersect([5], [5]) == [5]
