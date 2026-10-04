"""Command-line interface.

Commands (run from the project root, or `pip install -e .` first):

    PYTHONPATH=src python -m pir.cli build  data/cranfield --index-out index.json
    PYTHONPATH=src python -m pir.cli query  index.json '"information retrieval"' --explain
    PYTHONPATH=src python -m pir.cli query  index.json 'shock NEAR/3 wave' --rank bm25 --top 5
    PYTHONPATH=src python -m pir.cli query  index.json 'shock NEAR/3 wave' --why-doc D7
    PYTHONPATH=src python -m pir.cli stats  index.json
    PYTHONPATH=src python -m pir.cli demo
    PYTHONPATH=src python -m pir.cli interactive index.json

`docs` for `build` is either a JSON file {name: text} or a directory that
contains cran.all.1400 (the Cranfield collection layout produced by
scripts/download_cranfield.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import datasets
from .biword import build_biword_index
from .evaluate import ndcg_at_k, precision_at_k, recall_at_k
from .explain import explain_hit, why_not
from .index import PositionalIndex, build_index
from .phrase import candidate_docs, phrase_query
from .proximity import near_query, window_query
from .query_parser import QueryParseError, parse
from .rank import BM25, rank_bm25, rank_bm25_proximity
from .tokenizer import TokenizerOptions, tokenize


def load_docs(path: str) -> dict[str, str]:
    """Load {name: text} from a JSON file or a Cranfield directory."""
    if os.path.isdir(path):
        docs, _q, _r = datasets.load_cranfield(path)
        return docs
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index(path: str) -> PositionalIndex:
    if path.endswith(".json"):
        return PositionalIndex.load_json(path)
    if path.endswith(".pkl"):
        return PositionalIndex.load_pickle(path)
    if path.endswith(".pirz"):
        return PositionalIndex.load_compressed(path)
    raise SystemExit(f"unsupported index format: {path} (use .json, .pkl or .pirz)")


def execute_query(index: PositionalIndex, query):
    """Run a parsed Query against the index.

    Returns (matches, plan) where matches is
        dict doc_id -> payload (start positions / pairs / windows / None)
    and plan is a human-readable query-plan line (rarest-first, df values).
    """
    terms = query.terms
    missing = [t for t in set(terms) if not index.has_term(t)]
    if query.kind == "and":
        plan = " | ".join(f"{t}(df={index.df(t)})" for t in sorted(set(terms), key=index.df))
        if missing:
            return {}, plan + f" | missing: {missing}"
        docs = candidate_docs(index, terms)
        return {d: None for d in docs}, plan + f" | docs after intersection: {len(docs)}"

    if missing:
        tf_msg = ", ".join(repr(t) for t in missing)
        return {}, f"term(s) not in index: {tf_msg} -> 0 matches"

    plan_terms = sorted(set(terms), key=index.df)
    plan = " -> ".join(f"{t}(df={index.df(t)})" for t in plan_terms)
    inter = candidate_docs(index, terms)
    plan += f" | docs after intersection: {len(inter)}"

    if query.kind == "phrase":
        return phrase_query(index, terms), plan
    if query.kind == "near":
        return near_query(index, terms[0], terms[1], query.k, query.ordered), plan
    if query.kind == "window":
        return window_query(index, terms, query.k), plan
    raise ValueError(f"unknown query kind {query.kind}")


def rank_matches(index: PositionalIndex, query, matches: dict, rank_mode: str):
    """Order matched docs. Boolean sets are ranked by BM25 (documented)."""
    docs = list(matches)
    if rank_mode == "none" or not docs:
        return docs
    if rank_mode == "bm25":
        return [d for d, _ in rank_bm25(index, docs, query.terms)]
    if rank_mode == "bm25prox":
        return [d for d, _ in rank_bm25_proximity(index, docs, query.terms)]
    raise SystemExit(f"unknown rank mode {rank_mode!r} (use none|bm25|bm25prox)")


def format_query_header(index: PositionalIndex, query, plan: str) -> list[str]:
    lines = []
    if query.kind == "phrase":
        head = f'PHRASE QUERY: "{" ".join(query.terms)}"'
    elif query.kind == "near":
        mode = "ordered" if query.ordered else "unordered"
        head = f"PROXIMITY QUERY: {query.describe()}  ({mode})"
    elif query.kind == "window":
        head = f"WINDOW QUERY: {query.describe()}"
    else:
        head = f"BOOLEAN AND QUERY: {query.describe()}"
    lines.append(head)
    if plan:
        lines.append(f"Plan: {plan}")
    return lines


def run_query(index: PositionalIndex, query_text: str, *, explain: bool = False,
              rank_mode: str = "none", top: int | None = None,
              why_doc: str | None = None) -> str:
    """Run one query and return the formatted report as text."""
    try:
        query = parse(query_text, index.tokenizer_opts)
    except QueryParseError as e:
        return f"PARSE ERROR: {e}"

    t0 = time.perf_counter()
    matches, plan = execute_query(index, query)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    lines = format_query_header(index, query, plan)
    lines.append(f"Matching documents: {len(matches)}   (query time {elapsed_ms:.2f} ms)")

    ordered = rank_matches(index, query, matches, rank_mode)
    if top is not None:
        ordered = ordered[:top]
    for d in ordered:
        if explain:
            lines.append("  " + explain_hit(index, query, d, matches[d]))
        else:
            lines.append(f"  {index.name(d)}")

    if why_doc:
        lines.append(why_not(index, query, why_doc))
    if not ordered and not matches:
        lines.append('  (tip: use --why-doc <docname> to see how close it got)')
    return "\n".join(lines)


# ------------------------------------------------------------------ commands


def cmd_build(args) -> int:
    docs = load_docs(args.docs)
    index = build_index(docs, keep_text=not args.no_text)
    st = index.stats
    print(
        f"Indexed {st.documents} documents | {st.tokens} tokens | "
        f"{st.unique_terms} unique terms | avg doc len {st.avg_doc_len:.1f} | "
        f"build {st.build_seconds:.2f}s"
    )
    out = args.index_out or "index.json"
    if out.endswith(".json"):
        index.save_json(out)
    elif out.endswith(".pkl"):
        index.save_pickle(out)
    elif out.endswith(".pirz"):
        index.save_compressed(out)
    else:
        raise SystemExit("--index-out must end in .json, .pkl or .pirz")
    print(f"Index saved to {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
    return 0


def cmd_stats(args) -> int:
    index = load_index(args.index)
    st = index.stats
    n_docs = len(index.doc_len)
    print("=" * 40)
    print("POSITIONAL INVERTED INDEX — STATS")
    print("=" * 40)
    print(f"Documents        : {n_docs}")
    print(f"Tokens           : {index.total_tokens()}")
    print(f"Unique terms     : {len(index.postings)}")
    print(f"Avg doc length   : {index.avg_doc_len():.2f} tokens")
    if st:
        print(f"Build time       : {st.build_seconds:.2f}s")
    print(f"Tokenizer        : {index.tokenizer_opts.as_dict()}")
    top = sorted(index.postings.items(), key=lambda kv: -len(kv[1]))[:10]
    print("Top 10 terms by df:")
    for t, docs in top:
        print(f"  {t:<15} df={len(docs)}")
    return 0


def cmd_query(args) -> int:
    index = load_index(args.index)
    print(run_query(
        index, args.query, explain=args.explain, rank_mode=args.rank, top=args.top,
        why_doc=args.why_doc,
    ))
    return 0


def cmd_interactive(args) -> int:
    index = load_index(args.index)
    print("Interactive mode. Syntax: \"phrase\" | a NEAR/3 b | a ONEAR/3 b | "
          "WINDOW/10(a b c) | bare terms = AND.  :q to quit")
    while True:
        try:
            line = input("pir> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in (":q", ":quit", "exit"):
            break
        if not line:
            continue
        print(run_query(index, line, explain=True, rank_mode=args.rank, top=args.top))
    return 0


def cmd_demo(_args) -> int:
    """End-to-end demo on the toy corpus (the eyeball-verifiable set)."""
    docs = datasets.load_toy()
    index = build_index(docs)
    queries = [
        '"information retrieval"',
        '"king of spain"',
        '"to be or not to be"',
        '"new new york"',
        '"buffalo buffalo"',
        'data NEAR/2 data',
        'shock NEAR/3 wave',
        'shock ONEAR/3 wave',
        'heat NEAR/2 transfer',
        'WINDOW/4(x ray machine)',
        '"information retrieval" --why-doc d9',
        'heat NEAR/2 temperature --why-doc d9',
    ]
    print("=" * 40)
    print("POSITIONAL INFORMATION RETRIEVAL SYSTEM")
    print("=" * 40)
    st = index.stats
    print(
        f"Dataset: toy | Documents: {st.documents} | Tokens: {st.tokens} | "
        f"Unique terms: {st.unique_terms}"
    )
    for q in queries:
        why = None
        if "--why-doc" in q:
            q, why = [p.strip() for p in q.split("--why-doc")]
        print("-" * 40)
        print(run_query(index, q, explain=True, why_doc=why))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pir", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build an index from a JSON docs file or Cranfield dir")
    b.add_argument("docs")
    b.add_argument("--index-out", default="index.json")
    b.add_argument("--no-text", action="store_true", help="do not store raw texts")
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query", help="run one query against a saved index")
    q.add_argument("index")
    q.add_argument("query")
    q.add_argument("--explain", action="store_true", help="show positions + snippets")
    q.add_argument("--rank", default="none", choices=["none", "bm25", "bm25prox"])
    q.add_argument("--top", type=int, default=None)
    q.add_argument("--why-doc", default=None, help="explain why this doc did NOT match")
    q.set_defaults(func=cmd_query)

    s = sub.add_parser("stats", help="print index statistics")
    s.add_argument("index")
    s.set_defaults(func=cmd_stats)

    i = sub.add_parser("interactive", help="REPL")
    i.add_argument("index")
    i.add_argument("--rank", default="none", choices=["none", "bm25", "bm25prox"])
    i.add_argument("--top", type=int, default=10)
    i.set_defaults(func=cmd_interactive)

    d = sub.add_parser("demo", help="end-to-end demo on the toy corpus")
    d.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
