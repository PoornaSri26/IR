"""Cranfield loader tests: field parsing, qrels formats, relevance conversion.

These use tiny synthetic inputs, so no download is needed."""

from pir.datasets import (
    binarize_rel,
    canon_qid,
    load_cranfield,
    normalize_cranfield_rel,
    parse_cranfield_docs,
    parse_cranfield_qrels,
    parse_cranfield_queries,
)

DOCS = """.I 1
.T
Boundary layer theory
.W
The boundary layer grows with distance.
.A
Smith
.I 2
.W
Heat transfer coefficients.

.W
Second paragraph of doc 2."""

QUERIES = """.I 001
.W
what is a boundary layer
.I 002
.W
heat transfer coefficient"""

QRELS_3COL = """1 184 2
1 29 1
2 31 3"""

QRELS_4COL = """1 0 184 2
1 0 29 -1"""


def test_doc_fields_including_marker_on_own_line():
    docs = parse_cranfield_docs(DOCS)
    assert set(docs) == {"1", "2"}
    assert docs["1"].startswith("Boundary layer theory")
    assert "boundary layer grows" in docs["1"]
    # .W with no text on the marker line must still capture the next lines
    assert docs["2"] == "Heat transfer coefficients. Second paragraph of doc 2."


def test_queries_preserve_sparse_ids_at_parse_time():
    queries = parse_cranfield_queries(QUERIES)
    assert set(queries) == {"001", "002"}
    assert queries["001"] == "what is a boundary layer"


def test_canonical_qid_bridges_padding():
    # cranqrel keys may be padded; canonical form strips it
    assert canon_qid("001") == canon_qid("1") == "1"


SPARSE_QRY = """.I 001
.W
first query
.I 002
.W
second query
.I 008
.W
third query
.I 010
.W
fourth query"""


def test_queries_keyed_by_file_position_not_printed_id(tmp_path):
    """Regression test for the Cranfield query-numbering trap.

    cran.qry prints sparse ORIGINAL ids (001, 002, 008, 010) while cranqrel
    numbers the same queries 1..225 by their position in the file.  Joining on
    the printed id mis-aligns almost every query and silently zeroes the
    evaluation, so queries must be keyed by position.
    """
    d = tmp_path / "cranfield"
    d.mkdir()
    (d / "cran.all.1400").write_text(DOCS, encoding="latin-1")
    (d / "cran.qry").write_text(SPARSE_QRY, encoding="latin-1")
    (d / "cranqrel").write_text("1 1 1\n2 2 1\n3 1 2\n4 2 2\n", encoding="latin-1")

    docs, queries, qrels = load_cranfield(str(d))
    assert list(queries) == ["1", "2", "3", "4"]  # positional keys, contiguous
    assert queries["3"] == "third query"  # was printed as .I 008
    assert queries["4"] == "fourth query"  # was printed as .I 010
    assert set(qrels) == {"1", "2", "3", "4"}
    # query 3 is judged against doc 1: the join actually resolves
    assert qrels["3"] == {"1": 2}


def test_qrels_three_and_four_columns():
    q3 = parse_cranfield_qrels(QRELS_3COL)
    assert q3 == {"1": {"184": 2, "29": 1}, "2": {"31": 3}}
    q4 = parse_cranfield_qrels(QRELS_4COL)
    assert q4 == {"1": {"184": 2, "29": -1}}


def test_relevance_scale_is_inverted_then_normalized():
    # cranqrel.readme (Cleverdon): 1 = complete answer, 4 = minimum interest,
    # 5 = no interest (absent from the file), -1 = no judgement
    assert normalize_cranfield_rel(1) == 4  # most relevant after normalization
    assert normalize_cranfield_rel(4) == 1  # least relevant after normalization
    assert normalize_cranfield_rel(-1) == 0
    assert normalize_cranfield_rel(5) == 0
    # higher std grade must mean more relevant
    assert normalize_cranfield_rel(1) > normalize_cranfield_rel(3)
    # binarization at threshold 2 == raw value in {1, 2, 3}
    assert binarize_rel(normalize_cranfield_rel(1))
    assert binarize_rel(normalize_cranfield_rel(3))
    assert not binarize_rel(normalize_cranfield_rel(4))
    assert not binarize_rel(normalize_cranfield_rel(-1))