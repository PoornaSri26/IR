#!/usr/bin/env python
"""Mine frequent phrases from the collection (Level 3F).

Problem this solves: Cranfield's queries are natural-language sentences, not
phrase queries, so there is no ready-made phrase-query test set.  Mining
frequent bigrams/trigrams by raw frequency and by PMI gives a realistic set
of phrase queries whose expected behaviour (which docs contain them) is
verifiable by construction.

PMI(w1,w2) = log( p(w1,w2) / (p(w1) * p(w2)) )     (positive = the two words
stick together more than chance).  We report both ranking views because they
select different phrases: frequency finds common boilerplate, PMI finds
unusually cohesive pairs.

Usage:  python scripts/mine_phrases.py [--top 40] [--out results/mined_phrases.md]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pir.datasets import load_cranfield  # noqa: E402
from pir.tokenizer import tokenize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mine_phrases(docs: dict[str, str], n: int = 2, min_count: int = 5) -> list[tuple[str, int, float]]:
    """Return [(phrase, count, pmi)] for all n-grams with count >= min_count."""
    unigrams: Counter[str] = Counter()
    ngrams: Counter[str] = Counter()
    total_tokens = 0
    total_ngrams = 0

    for text in docs.values():
        toks = tokenize(text)
        unigrams.update(toks)
        total_tokens += len(toks)
        for i in range(len(toks) - n + 1):
            ngrams[" ".join(toks[i : i + n])] += 1
            total_ngrams += 1

    out = []
    for phrase, c in ngrams.items():
        if c < min_count:
            continue
        words = phrase.split()
        p_ab = c / total_ngrams
        p_a = unigrams[words[0]] / total_tokens
        p_b = unigrams[words[1]] / total_tokens
        pmi = math.log(p_ab / (p_a * p_b)) if p_a and p_b else 0.0
        out.append((phrase, c, pmi))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mined_phrases.md"))
    args = ap.parse_args()

    cran_dir = os.path.join(ROOT, "data", "cranfield")
    if not os.path.exists(os.path.join(cran_dir, "cran.all.1400")):
        print("Cranfield not found — run: python scripts/download_cranfield.py", file=sys.stderr)
        return 1
    docs, _q, _r = load_cranfield(cran_dir)

    lines = ["# Mined phrases (Level 3F)", ""]
    lines.append(f"Source: Cranfield ({len(docs)} documents), tokenizer = default options.")
    lines.append("")
    for n in (2, 3):
        phrases = mine_phrases(docs, n=n, min_count=args.min_count)
        by_freq = sorted(phrases, key=lambda x: (-x[1], x[0]))[: args.top]
        by_pmi = sorted(phrases, key=lambda x: (-x[2], -x[1]))[: args.top]
        lines.append(f"## {n}-grams (count >= {args.min_count})")
        lines.append("")
        lines.append(f"### Top {len(by_freq)} by raw frequency")
        lines.append("")
        lines.append("| phrase | count | PMI |")
        lines.append("|---|---:|---:|")
        for phrase, c, pmi in by_freq:
            lines.append(f"| {phrase} | {c} | {pmi:.2f} |")
        lines.append("")
        lines.append(f"### Top {len(by_pmi)} by PMI (most cohesive pairs)")
        lines.append("")
        lines.append("| phrase | count | PMI |")
        lines.append("|---|---:|---:|")
        for phrase, c, pmi in by_pmi:
            lines.append(f"| {phrase} | {c} | {pmi:.2f} |")
        lines.append("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())