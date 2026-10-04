"""Dataset loaders: toy corpus (hand-made, 10 docs) and Cranfield.

Cranfield collection (1,400 aeronautics abstracts, 225 queries, graded
qrels) — see README section 3 for source, license and download steps, and
scripts/download_cranfield.py for the automated fetch.

cran.all.1400 fields: .I id / .T title / .A author(s) / .B bibliography /
.W abstract.  We index title + abstract (T + W), documented in the README.

cranqrel relevance values (from the collection README, Glasgow IR group):
    -1 = no relevance
     1 = complete answer          (MOST relevant)
     2 = high degree of relevance
     3 = useful as background / suggests methods
     4 = minimum interest
     5 = no interest              (never appears in the qrels)
i.e. the RAW scale runs BACKWARDS (1 is best, -1/5 mean not relevant).
`normalize_cranfield_rel` converts to the standard direction
(4 = most relevant, 0 = not relevant) and this is unit-tested so nobody
gets bitten by the trap the collection README warns about.

See `load_cranfield` for the SECOND trap: cran.qry prints sparse original
query numbers while cranqrel numbers queries 1..225 by file position.
"""

from __future__ import annotations

import json
import os

from .tokenizer import STOPWORDS, tokenize

# --------------------------------------------------------------------- toy


def load_toy(path: str | None = None) -> dict[str, str]:
    if path is None:
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "toy", "toy_docs.json"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- cranfield


def parse_cranfield_docs(text: str) -> dict[str, str]:
    """Parse cran.all.1400 -> {doc_id_str: title + ' ' + abstract}."""
    docs: dict[str, str] = {}
    cur_id: str | None = None
    cur_field: str | None = None
    fields: dict[str, list[str]] = {}

    def flush():
        nonlocal fields
        if cur_id is not None:
            title = " ".join(x for x in fields.get("T", []) if x.strip())
            abstract = " ".join(x for x in fields.get("W", []) if x.strip())
            docs[cur_id] = (title + " " + abstract).strip()

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("."):
            tag = line[1].upper()
            if tag == "I":
                flush()
                cur_id = line[2:].strip()
                cur_field = None
                fields = {}
                continue
            if tag in ("T", "A", "B", "W"):
                cur_field = tag
                fields.setdefault(cur_field, [])  # marker may be on its own line
                rest = line[2:].strip()
                if rest:
                    fields[cur_field].append(rest)
                continue
        if cur_field is not None and cur_id is not None:
            fields[cur_field].append(line.strip())
    flush()
    return docs


def parse_cranfield_queries(text: str) -> dict[str, str]:
    """Parse cran.qry -> {query_id_str: query_text}."""
    queries: dict[str, str] = {}
    cur_id: str | None = None
    cur_field: str | None = None
    buf: list[str] = []

    def add(text: str):
        if text:
            buf.append(text)

    def flush():
        if cur_id is not None and buf:
            queries[cur_id] = " ".join(x for x in buf if x).strip()

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith(".I"):
            flush()
            cur_id = line[2:].strip()
            cur_field = None
            buf = []
            continue
        if line.startswith(".W"):
            cur_field = "W"
            add(line[2:].strip())
            continue
        if cur_id is not None:
            add(line.strip())
    flush()
    return queries


def parse_cranfield_qrels(text: str) -> dict[str, dict[str, int]]:
    """Parse cranqrel -> {qid: {doc_id: raw_rel}}.

    Lines may have 3 fields (qid doc rel) or 4 (qid iter doc rel, TREC
    style); the loader handles both and reports nothing else.
    """
    qrels: dict[str, dict[str, int]] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        if len(parts) == 4:
            qid, _it, doc, rel = parts
        elif len(parts) == 3:
            qid, doc, rel = parts
        else:
            continue
        try:
            r = int(rel)
        except ValueError:
            continue
        qrels.setdefault(qid, {})[doc] = r
    return qrels


def normalize_cranfield_rel(raw: int) -> int:
    """Convert the RAW cranqrel scale to the STANDARD direction.

    raw: -1 -> 0 (not relevant), 1 (high) -> 4, 2 -> 3, 3 -> 2, 4 -> 1,
    5 ("unrelevant") -> 0.
    """
    if raw == -1 or raw == 5:
        return 0
    if 1 <= raw <= 4:
        return 5 - raw
    return 0


def binarize_rel(std_rel: int, threshold: int = 2) -> bool:
    """A document counts as relevant if std_rel >= threshold (default 2 =
    'possibly relevant' or better).  Documented in README section 13."""
    return std_rel >= threshold


def canon_qid(qid: str) -> str:
    """Canonical (unpadded) query-id form, used for cranqrel keys."""
    try:
        return str(int(str(qid).strip()))
    except ValueError:
        return str(qid).strip()


def load_cranfield(dirpath: str):
    """Returns (docs, queries, qrels_raw).

    docs:    {doc_id_str: text}
    queries: {qid_str: text}              qid = POSITION (1..225) in cran.qry
    qrels_raw: {qid: {doc_id: RAW relevance}}  qid = cranqrel's 1..225 numbering

    *** The query-numbering trap (this one silently destroys evaluation) ***
    cran.qry carries the ORIGINAL Cranfield query numbers, which are a SPARSE
    subset of the old 416-query collection: the file order of ids is
    001, 002, 004, 008, 009, 010, 012, 013, 015, 018, 022, 023, ...
    cranqrel, by contrast, numbers the SAME 225 queries 1..225 CONTIGUOUSLY.
    Joining queries to qrels on the printed id therefore mis-aligns almost
    every query: only queries 1 and 2 happen to coincide.

    Worked example: cran.qry id "008" (4th in file) reads "can a criterion be
    developed to show empirically the validity of flow solutions for
    chemically reacting gas mixtures".  cranqrel query 4 judges document 166,
    whose title is "flow of chemically reacting gas mixtures" -- clearly a
    match.  Joining on the printed id "8" would have looked that judgement up
    under the completely different query 8 ("heat conduction in composite
    slabs"), and evaluation returns near-zero scores for every system.

    So: queries are keyed by FILE POSITION, which is what the qrels mean.
    """
    with open(os.path.join(dirpath, "cran.all.1400"), "r", encoding="latin-1") as f:
        docs = parse_cranfield_docs(f.read())
    with open(os.path.join(dirpath, "cran.qry"), "r", encoding="latin-1") as f:
        queries_raw = parse_cranfield_queries(f.read())
    # dict preserves insertion order == file order
    queries = {str(pos): text for pos, (_, text) in enumerate(queries_raw.items(), start=1)}
    qrels = {}
    qrel_path = os.path.join(dirpath, "cranqrel")
    if not os.path.exists(qrel_path):
        qrel_path = os.path.join(dirpath, "cranqrel.tsv")
    if os.path.exists(qrel_path):
        with open(qrel_path, "r", encoding="latin-1") as f:
            qrels_raw = parse_cranfield_qrels(f.read())
        qrels = {canon_qid(q): v for q, v in qrels_raw.items()}
    return docs, queries, qrels


# ----------------------------------------------- query derivation (7.2 opt 1)


def content_terms(query_text: str, opts=None) -> list[str]:
    """Query tokens minus stopwords -> content words for derived queries."""
    return [t for t in tokenize(query_text, opts) if t not in STOPWORDS]


def derive_queries(query_text: str, opts=None) -> dict[str, list]:
    """Derive structured queries from a natural-language Cranfield query.

    Documented rule (README section 13):
    - and:    all content words
    - phrase: every adjacent content-word bigram (and the trigram when all
              three consecutive words are content words)
    - near:   every adjacent content-word pair, used with NEAR/k
    """
    toks = content_terms(query_text, opts)
    phrases: list[list[str]] = []
    for i in range(len(toks) - 1):
        phrases.append([toks[i], toks[i + 1]])
    for i in range(len(toks) - 2):
        phrases.append([toks[i], toks[i + 1], toks[i + 2]])
    pairs: list[tuple[str, str]] = [(toks[i], toks[i + 1]) for i in range(len(toks) - 1)]
    return {"and": toks, "phrase": phrases, "near": pairs}
