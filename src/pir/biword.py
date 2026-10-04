"""Biword index (Level 3E comparison): every consecutive token pair.

Used only for the mini-study that justifies the positional index:
- a 2-word phrase is a single dictionary lookup (faster than positional),
- the index is ~as large as the positional one again (every bigram stored),
- n >= 3 phrases need biword decomposition + doc-level intersection, which
  produces FALSE POSITIVES when the overlapping bigrams occur at different
  positions (measured empirically in scripts/run_experiments.py).
"""

from __future__ import annotations

from .index import PositionalIndex, build_index, natural_key
from .tokenizer import TokenizerOptions, tokenize


class BiwordIndex:
    def __init__(self, postings: dict[str, dict[int, list[int]]], doc_names: dict[int, str]):
        self.postings = postings  # "w1 w2" -> {doc_id: [start positions]}
        self.doc_names = doc_names

    def df(self, bigram: str) -> int:
        return len(self.postings.get(bigram, ()))


def build_biword_index(docs: dict[str, str], tokenizer_opts: TokenizerOptions | None = None) -> BiwordIndex:
    opts = tokenizer_opts or TokenizerOptions()
    postings: dict[str, dict[int, list[int]]] = {}
    doc_names: dict[int, str] = {}
    for doc_id, name in enumerate(sorted(docs, key=natural_key), start=1):
        doc_names[doc_id] = name
        toks = tokenize(docs[name], opts)
        for i in range(len(toks) - 1):
            bg = f"{toks[i]} {toks[i + 1]}"
            postings.setdefault(bg, {}).setdefault(doc_id, []).append(i + 1)
    return BiwordIndex(postings, doc_names)


def biword_phrase_query(bw: BiwordIndex, terms: list[str], positional: PositionalIndex | None = None):
    """Phrase query on the biword index.

    - n == 2: exact single lookup -> {doc: [start positions]}
    - n >= 3: intersect doc sets of all consecutive bigrams; when a
      `positional` index is given, false positives are filtered with the
      exact positional check (both numbers reported by the experiments).
    Returns {doc_id: [start positions]} (start positions exact only when
    `positional` is provided or n == 2).
    """
    if len(terms) < 2:
        raise ValueError("biword phrase query needs >= 2 terms")
    bigrams = [f"{terms[i]} {terms[i + 1]}" for i in range(len(terms) - 1)]
    if any(bg not in bw.postings for bg in bigrams):
        return {}

    if len(terms) == 2:
        bg = bigrams[0]
        return {d: list(ps) for d, ps in bw.postings[bg].items()}

    doc_sets = [set(bw.postings[bg]) for bg in bigrams]
    cur = doc_sets[0]
    for s in doc_sets[1:]:
        cur &= s
        if not cur:
            return {}
    result = {d: [] for d in cur}
    if positional is not None:
        from .phrase import phrase_query

        exact = phrase_query(positional, terms)
        return {d: exact.get(d, []) for d in exact}
    return result
