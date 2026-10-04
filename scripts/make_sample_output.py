#!/usr/bin/env python
"""Generate output/sample_output.txt — the demo transcript for the README.

Shows the header stats, phrase queries with positions + highlighted snippets,
ordered and unordered proximity, a window query, ranked BM25 output, and the
explain-why-not diagnostics for documents that just missed.

Usage:  python scripts/make_sample_output.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from pir.cli import run_query  # noqa: E402
from pir.compress import encode_postings_varbyte  # noqa: E402
from pir.datasets import load_cranfield, load_toy  # noqa: E402
from pir.explain import why_not  # noqa: E402
from pir.index import build_index  # noqa: E402
from pir.proximity import min_distance  # noqa: E402
from pir.query_parser import parse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAN = os.path.join(ROOT, "data", "cranfield")
OUT = os.path.join(ROOT, "output", "sample_output.txt")

BAR = "-" * 40


def find_near_miss(index, t1, t2, k):
    """A document holding both terms but farther apart than k (for why-not)."""
    from pir.phrase import candidate_docs

    for d in candidate_docs(index, [t1, t2]):
        res = min_distance(index, t1, t2, d)
        if res and res[0] > k:
            return index.doc_names[d]
    return None


def find_has_one_term(index, t1, t2):
    from pir.phrase import candidate_docs

    for d in index.docs(t1):
        if not index.tf(t2, d):
            return index.doc_names[d]
    return None


def main() -> int:
    lines = []
    if not os.path.exists(os.path.join(CRAN, "cran.all.1400")):
        print("Cranfield not found — run: python scripts/download_cranfield.py", file=sys.stderr)
        return 1
    docs, queries, qrels = load_cranfield(CRAN)
    index = build_index(docs)
    st = index.stats

    raw_bytes = len(__import__("pickle").dumps(index.postings, protocol=5))
    comp_bytes = len(encode_postings_varbyte(index.postings))

    lines += [
        "=" * 40,
        "POSITIONAL INFORMATION RETRIEVAL SYSTEM",
        "=" * 40,
        f"Dataset: Cranfield | Documents: {st.documents} | Tokens: {st.tokens} | "
        f"Unique terms: {st.unique_terms}",
        f"Index built in {st.build_seconds:.2f}s | Size: postings raw {raw_bytes/1e6:.2f} MB, "
        f"delta+varbyte {comp_bytes/1e6:.2f} MB",
        f"Positions are 1-based | distance convention: adjacent words are distance 1",
        "",
    ]

    demo_queries = [
        ('"boundary layer"', True, None),
        ('"heat transfer"', True, None),
        ('"boundary layer" --why-doc', False, find_has_one_term(index, "boundary", "layer")),
        ("shock NEAR/3 wave", True, None),
        ("shock ONEAR/3 wave", True, None),
        ("shock NEAR/3 wave --why-doc", False, find_near_miss(index, "shock", "wave", 3)),
        ("WINDOW/12(supersonic boundary layer interaction)", True, None),
        ("aeroelastic flutter theory", False, None),
        ("mach number NEAR/1 supersonic", True, None),
    ]

    for q, explain, why in demo_queries:
        if q.endswith("--why-doc"):
            q = q[: -len("--why-doc")].strip()
        lines.append(BAR)
        lines.append(
            run_query(
                index, q,
                explain=explain,
                rank_mode="bm25" if q == "aeroelastic flutter theory" else "none",
                top=5 if q == "aeroelastic flutter theory" else 3,
                why_doc=why,
            )
        )
        lines.append("")

    # BM25 ranked list, shown with the score
    lines.append(BAR)
    lines.append("RANKED QUERY (BM25 over all 1400 documents): aeroelastic flutter theory")
    from pir.rank import rank_bm25

    ranked = rank_bm25(index, sorted(index.doc_len), ["aeroelastic", "flutter", "theory"])
    for d, score in ranked[:5]:
        lines.append(f"  {index.name(d):>6}  score={score:6.3f}  {index.positions('flutter', d)[:6]} <- 'flutter' positions")
    lines.append("")

    # toy corpus section (eyeball-verifiable edge cases)
    lines.append("=" * 40)
    lines.append("TOY CORPUS (10 documents) — edge cases, verifiable by eye")
    lines.append("=" * 40)
    toy = build_index(load_toy())
    lines.append(
        f"Documents: {toy.stats.documents} | Tokens: {toy.stats.tokens} | "
        f"Unique terms: {toy.stats.unique_terms}"
    )
    toy_cases = [
        ('"king of spain"', None),          # needs stopwords in the index
        ('"to be or not to be"', None),
        ('"new new york"', None),           # repeated term inside a phrase
        ('"buffalo buffalo"', None),        # repeated term inside a phrase
        ("data NEAR/2 data", None),         # same term, never paired with itself
        ("data NEAR/0 data", None),         # k = 0 is impossible
        ('WINDOW/4(x ray machine)', None),  # punctuation split into x / ray
        ('"across information retrieval"', None),  # would span d9 -> d10: must NOT match
    ]
    for q, why in toy_cases:
        lines.append(BAR)
        lines.append(run_query(toy, q, explain=True, why_doc=why))

    lines.append(BAR)
    lines.append(why_not(toy, parse('"information retrieval"'), "d9"))
    lines.append(why_not(toy, parse("heat NEAR/2 temperature"), "d9"))
    lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())