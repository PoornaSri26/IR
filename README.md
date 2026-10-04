# Positional Information Retrieval System

**A pure-IR system**: a positional inverted index over a document collection,
with phrase queries, proximity queries (ordered and unordered), explainable
output, and Cranfield evaluation.

Everything is implemented from scratch — the index, the tokeniser, the posting
list algorithms, compression, and the metrics.  No Lucene, Elasticsearch,
Whoosh, or other IR library is used for the index.  Stemming/POS/NER/embeddings
are deliberately absent (Level 1 boundary, section 1).

---

## 1. Objective

A positional inverted index whose correctness is *demonstrated* rather than
claimed:

- phrase queries of any length (offset-trick), with a **brute-force oracle**
  differential test on ~4,800 randomized queries (200 random corpora × 6
  queries × 4 query types);
- **ordered and unordered** NEAR/k/ONEAR/k proximity, with a two-pointer
  merge for detection and pair enumeration for explanation;
- **explainable output** (positions, highlighted snippets, and a
  why-not report: *"heat at 1, temperature at 6; min distance 5 > k=2"*), a
  Level 3G feature;
- **evaluation on Cranfield** (1,400 abstracts, 225 queries, 1,837 graded
  qrels) with P@k, MAP, nDCG@10, and the honest finding that phrase and
  proximity queries are high-precision/low-recall filters;
- **performance**: index build, size (raw vs δ+varbyte vs δ+gamma), and
  per-operation latency; skip-pointer intersection (1.25×, verified identical);
  biword-vs-positional mini-study (2-word phrases are ~21× faster, but 3-word
  phrases produce 23% false positives without a positional filter); and a
  tokenization ablation echoing the professor's published work on
  collection/normalisation choices.

## 2. Dataset

**Cranfield** (Cleverdon 1967), the classic small IR test collection:

| Item | Value |
|---|---|
| Documents | 1,400 aeronautics abstracts |
| Queries | 225 natural-language questions |
| Qrels | graded relevance, 1,837 judgements (raw values −1, 1..4) |

The collection ships as `cran.all.1400`, `cran.qry`, `cranqrel` (autodownloaded
by `scripts/download_cranfield.py` if absent; mirrors are included).  Fields
`.I/.T/.A/.B/.W` (title/abstract) are indexed as **title + abstract**; queries
are tokenized with the same pipeline.  License: public domain style (Glasgow
IR Group collection, Cleverdon 1967).

**The two traps of this collection** (documented in `datasets.py` and the
README, and handled correctly):

1. **Inverted relevance scale.**  In `cranqrel`, `1 = complete answer` (most
   relevant) rising to `4 = minimum interest`, `5 = none` (absent), `-1 =
   none`.  Higher means *less* relevant.  `normalize_cranfield_rel` turns it
   into the standard direction (4 = most relevant) and tests pin it.
2. **Sparse query numbers.**  `cran.qry` prints the *original* query numbers,
   which are a sparse subset of the old 416-query collection
   (`001, 002, 004, 008, 009, 010, 012, ...`), while `cranqrel` numbers the
   225 queries **1..225 by their order in the file**.  Joining queries and
   qrels on the printed id mis-aligns nearly every query (only 1 and 2
   coincide) and silently produces near-zero scores — this was discovered
   empirically during the project and cost days.  The loader therefore keys
   queries by **file position**; tests fail if it regresses.

## 3. Preprocessing

One tokeniser, shared by documents *and* queries (the classic bug source — a
query tokenised differently from the index):

- **Case folding on** (lowercased before matching);
- tokens = maximal runs of `[a-z0-9]`; punctuation splits tokens
  (`"don't"` → `don`/`t`, `"x-ray"` → `x`/`ray`, suffix `"3.5"` → `3`/`5`);
- **digits kept** by default;
- **stopwords kept** — removing them shifts positions and breaks phrases
  (`"king of spain"`, `"to be or not to be"`); stopword removal is an
  *ablation* option only;
- positions are **1-based** (the first token of a document is at 1); the base
  is arbitrary, only consistency matters;
- a toy stemmer exists for the ablation study only (section 8), never in the
  main system.

## 4. Positional inverted index

```
postings:  term -> {doc_id -> [pos1, pos2, ...]}   positions ascending, 1-based
doc_len:   doc_id -> token count
doc_names: internal int doc_id -> original name (e.g. "1400", "008")
```

- Build: one hash-map insert per token (O(total tokens)); deterministic doc
  ids (natural sort of original names).
- Persistence: JSON (portable), pickle (fast), and δ+variable-byte / δ+gamma
  (`.pirz`, Level 3A) — the compressed stores carry postings *and* doc
  metadata but not raw texts; the JSON/pickle carry texts too.
- Query-time helpers: `candidate_docs` (rarest term first, short-circuit),
  `intersect_docids` / `skip_intersect` (Level 3B, verified identical).
- Stats: documents, tokens, unique terms, average doc length, build time.

## 5. Phrase queries

`"information retrieval"` — any n ≥ 1:

1. Missing term → empty (no exception).
2. Candidate docs: intersection rarest-term first, short-circuit on empty.
3. Offset trick: intersect `positions(t1)` with `{p - i : p in positions(t_i)}`;
   correct for repeated terms (`"new new york"`, `"buffalo buffalo"`).
4. Positions are per-document → **never matches across a boundary**.

## 6. Proximity queries

Distance convention (stated everywhere, the #1 viva confusion): **adjacent
words have distance 1**; distance between two occurrences is `|p1 - p2|`.
Therefore:

- **unordered** `A NEAR/k B`: `p1 != p2` and `|p1 - p2| <= k`
- **ordered** `A ONEAR/k B`: `0 < p2 - p1 <= k` (B after A)
- **phrase `"a b"` == `a ONEAR/1 b`** exactly

Two-pointer merge for *detection* (O(|P1| + |P2|) per doc, plus a correct
equal-position branch so same-term operands never self-pair); full pair
enumeration via bisect windows (O(tf₁·log tf₂)) so no valid pair is ever
missed.  k = 0 is impossible (distinct positions differ by ≥ 1); k larger
than the document behaves like a Boolean AND.  Chains of the same operator +
same k (`a NEAR/3 b NEAR/3 c`) are accepted as a multi-term window.

## 7. Window queries (Level 3D)

`WINDOW/10(a b c)`: all terms inside a window of w tokens, defined as
`max(p) - min(p) <= w`.  A sliding-window minimal-cover over the merged
position stream finds all minimal windows in linear time; a term appearing m
times needs m *distinct* occurrences.  For two terms, `WINDOW/k(a b) ==
a NEAR/k b` (documented invariant, unit-tested).

## 8. Query language

| Syntax | Meaning |
|---|---|
| `"information retrieval"` | phrase (consecutive, in order) |
| `information NEAR/3 retrieval` | unordered `|p1-p2| <= 3` |
| `information ONEAR/3 retrieval` | ordered `0 < p2-p1 <= 3` |
| `information retrieval` | Boolean AND (rankable) |
| `WINDOW/10(a b c)` | all terms within a 10-token window |

- operators case-insensitive; extra whitespace tolerated; NEAR operands are
  single terms; a chain of the *same* operator + *same* k is a window;
- malformed queries raise `QueryParseError` with actionable messages
  (unbalanced quotes, `NEAR/` without a number, negative k, empty query, etc.).

## 9. Algorithms and complexity (documented per query)

- Index build: O(N) over total tokens (hash inserts, amortized O(1)).
- Doc intersection: O(|P₁| + |P₂|) with merge; O(√n) skip pointers (Level 3B).
- Phrase check per doc: O(Σ tf over query terms).
- Proximity detection: O(|P₁| + |P₂|) per doc via two-pointer merge.
- BM25 scoring: O(n·documents) per query; idf = ln(1 + (N−df+0.5)/(df+0.5)),
  k₁ = 1.2, b = 0.75 (Lucene-style, non-negative).

## 10. Project structure

```
positional-ir/
├── data/toy/toy_docs.json        # 10-doc hand-made corpus (edge cases)
├── data/cranfield/                 # cran.all.1400, cran.qry, cranqrel
├── scripts/
│   ├── download_cranfield.py       # autodownload + verify
│   ├── mine_phrases.py             # bigram/trigram mining by freq & PMI
│   ├── run_experiments.py          # all Cranfield experiments + reports
│   └── make_sample_output.py       # generates output/sample_output.txt
├── src/pir/
│   ├── tokenizer.py                # the single document/query tokeniser
│   ├── index.py                    # build, save/load, stats, skips
│   ├── compress.py                 # δ + variable-byte / Elias-gamma
│   ├── phrase.py, proximity.py     # executors
│   ├── window.py                   # WINDOW/w executor
│   ├── query_parser.py             # syntax + clear errors
│   ├── rank.py                     # BM25, BM25+proximity
│   ├── explain.py                  # snippets, positions, why-not
│   ├── evaluate.py                 # P@k, AP/MAP, nDCG (from scratch)
│   ├── datasets.py                 # toy + Cranfield loaders
│   ├── biword.py                   # biword index for the mini-study
│   └── cli.py                      # build/query/stats/interactive/demo
├── tests/                          # 100 tests: unit, oracle, metamorphic,
│                                   # round-trip, compression, significance
├── output/sample_output.txt
├── results/
│   ├── tables.md, systems.csv, ablation.csv, mined_phrases.md
│   ├── precision_recall_vs_k.png, map_by_system.png, index_size.png
│   └── idx.{json,pkl,pirz}         # the built index files (scratch leftovers
└── README.md
```

## 11. Installation and running

```bat
cd positional-ir
pip install -e .            # optional, makes `pir` importable anywhere
```

Without a global install, prefix every command with `PYTHONPATH=src`.
No runtime dependencies: pure Python 3.10+ standard library.

```bat
python -m pir.cli build  data/cranfield --index-out index.json
python -m pir.cli query  index.json '"boundary layer"' --explain
python -m pir.cli query  index.json 'shock NEAR/3 wave' --why-doc 18
python -m pir.cli query  index.json 'WINDOW/10(aeroelastic flutter theory)' --rank bm25 --top 5
python -m pir.cli stats  index.json
python -m pir.cli demo                 # end-to-end demo on the toy corpus
python -m pir.cli interactive index.json

python scripts/download_cranfield.py     # fill data/cranfield/ (1 step)
python scripts/run_experiments.py        # full Cranfield evaluation -> results/
python scripts/make_sample_output.py     # -> output/sample_output.txt
```

## 12. Sample output

(verbatim excerpt from `output/sample_output.txt`):

```
========================================
POSITIONAL INFORMATION RETRIEVAL SYSTEM
========================================
Dataset: Cranfield | Documents: 1400 | Tokens: 243204 | Unique terms: 7471
Index built in 0.53s | Size: postings raw 1.37 MB, delta+varbyte 0.61 MB
Positions are 1-based | distance convention: adjacent words are distance 1

----------------------------------------
PHRASE QUERY: "boundary layer"
Plan: layer(df=398) -> boundary(df=459) | docs after intersection: 359
Matching documents: 353   (query time 2.62 ms)
  1  boundary@111 layer@112
       ...ent produced by the slipstream was due to a /destalling/ or [boundary]-[layer]-control effect .  the integrated remaining lift increment, ...
  2  boundary@76 layer@77
       ...iscid rotational flow region between the shock wave and the [boundary] [layer] .  such a situation arises, for instance, in the study of t...
----------------------------------------
PROXIMITY QUERY: shock NEAR/3 wave  (unordered)
Plan: wave(df=168) -> shock(df=237) | docs after intersection: 117
Matching documents: 98   (query time 1.29 ms)
  2  shock@[50, 72] wave@[51, 73] (50,51) (72,73)
       ...mensional body it is usually necessary to consider a curved [shock] [wave] emitting from the nose or leading edge of the body . ...
WHY-NOT: 18 lacks the term 'layer' entirely

----------------------------------------
PROXIMITY QUERY: shock ONEAR/3 wave  (ordered)
Plan: wave(df=168) -> shock(df=237) | docs after intersection: 117
Matching documents: 98   (query time 1.13 ms)
  2  shock@[50, 72] wave@[51, 73] (50,51) (72,73)
       ...curved [shock] [wave] emitting from the nose or leading edge of the body . ...

========================================
TOY CORPUS (10 documents) — edge cases, verifiable by eye
========================================
Documents: 10 | Tokens: 136 | Unique terms: 84
----------------------------------------
PHRASE QUERY: "king of spain"
  d3  king@2 of@3 spain@4  ...The [king] [of] [Spain] lives in Madrid...
PHRASE QUERY: "new new york"   (repeated term)  ->  d5  new@1 new@2 york@3
PROXIMITY QUERY: data NEAR/2 data  ->  d6  data@[1,2] data@[2,1]  (1,2) (2,1)
PROXIMITY QUERY: data NEAR/0 data  ->  0 matches
PHRASE QUERY: "across information retrieval"  ->  0 matches  (no cross-boundary)
why-not: d9 lacks the term 'retrieval' entirely
why-not: d9 does not match heat NEAR/2 temperature: 'heat' at 1, 'temperature' at 6; min distance = 5 > k=2
```

Run `python -m pir.cli demo` for the same transcript interactively.

## 13. Evaluation (Cranfield, 225 queries)

All metrics computed from scratch; `results/tables.md` holds the full tables.

**Conventions.**  Binary relevant = std grade ≥ 2 (raw index value 1-3;
see section 2).  Constrained systems return the *union* of their derived
sub-query matches and are ranked by BM25 inside that set, so P@k / MAP /
nDCG are comparable across systems; a query whose candidate set is empty
scores AP = 0 (honest low recall, counted in the "empty queries" column).

| System | P@5 | P@10 | R@10 | MAP | nDCG@10 | set-recall | set-precision | empty queries |
|---|---|---|---|---|---|---|---|---|
| BM25 | 0.2640 | 0.1907 | 0.4067 | **0.2748** | **0.3446** | 0.987 | 0.004 | 0/225 |
| BM25 + proximity | 0.2640 | 0.1911 | 0.4076 | 0.2743 | 0.3443 | 0.987 | 0.004 | 0/225 |
| Boolean AND | 0.0062 | 0.0031 | 0.0063 | 0.0057 | 0.0074 | 0.006 | 0.010 | **218/225** |
| Phrase | 0.2329 | 0.1569 | 0.3225 | 0.2138 | 0.2931 | 0.476 | **0.081** | 3/225 |
| NEAR/1 | 0.2356 | 0.1587 | 0.3265 | 0.2160 | 0.2963 | 0.482 | 0.078 | 3/225 |
| NEAR/2 | 0.2453 | 0.1644 | 0.3441 | 0.2256 | 0.3045 | 0.518 | 0.070 | 1/225 |
| NEAR/3 | 0.2480 | 0.1702 | 0.3594 | 0.2308 | 0.3111 | 0.547 | 0.069 | 0/225 |
| NEAR/5 | 0.2498 | 0.1729 | 0.3644 | 0.2354 | 0.3157 | 0.561 | 0.064 | 0/225 |
| NEAR/10 | 0.2524 | 0.1751 | 0.3707 | 0.2393 | 0.3190 | 0.587 | 0.059 | 0/225 |
| NEAR/25 | 0.2542 | 0.1756 | 0.3748 | 0.2430 | 0.3226 | 0.612 | 0.048 | 0/225 |

**Honest finding.**  The unconstrained BM25 keeps full recall and wins on
MAP/nDCG.  Every constrained system gives up recall: phrase and proximity
return only ~half of the relevant documents (set-recall 0.48→0.61 as k grows),
but concentrate them — set-precision rises from 0.05 (k=25) to 0.08 (k=1).
This is exactly the textbook behaviour of phrase/proximity queries as
*high-precision, low-recall precision tools*, and it is why they must be
combined with a ranked baseline rather than used alone.  Paired bootstrap
per-query AP tests (10,000 resamples): BM25 vs BM25+proximity p = 0.30
(not significant), BM25 vs Phrase p < 0.0001.

**k sweep:** both the set metrics show the classic trade-off — as k shrinks,
set-precision rises (0.048 → 0.078) and set-recall falls (0.61 → 0.48)
— while ranked P@10 is almost flat because the BM25 ranking inside each
constrained set is untouched by k.

**k sensitivity:** NEAR/25 already reaches 99% of the AND recall
(0.612 vs the Boolean AND set recall 0.619), showing that *some* co-location
filter recovers almost nothing that a broad BM25 misses — precision, not
recall, is the only thing proximity buys.

**Efficiency.**

| quantity | value |
|---|---|
| index build time (3 runs) | 0.55 s |
| postings entries | 122,884 |
| positions | 243,204 |
| varbyte size | 0.61 MB (2.23× smaller than raw pickle) |
| gamma size | 0.55 MB (2.47× smaller) |
| decode cost | varbyte 0.52 ms, gamma 1.50 ms vs 0.22 ms pickle round-trip |
| BM25 top-10 per query | 66.6 ms |
| phrase executor (per derived phrase) | 0.17 ms |
| NEAR/3 (per derived pair) | 0.20 ms |
| skip-pointer speedup | 1.25× (identical results) |
| 2-word phrase: biword vs positional | **21× faster** (21.4 ms vs 455 ms) |
| 3-word phrase: biword false-positive rate | **23%** (removed by positional filter, 463 ms) |

## 14. Testing

Run with `pytest -q` (100 tests, ~16 s):

1. **Oracle/differential tests** (~4,800 randomized queries): every query type
   recomputed from raw token lists on 200 random corpora with repeated terms,
   k = 0..7, w = 0..8 — index results must equal brute-force results.
2. **Metamorphic properties**: ONEAR/1 == phrase (2 terms); NEAR/k ⊆ NEAR/k+1;
   ordered ⊆ unordered; phrase ⊆ AND; phrase ⊆ ONEAR/1.
3. Unit edge cases: term missing, empty query, single-term phrase, repeated
   terms, no cross-boundary, self-pair exclusion (`data NEAR/2 data`), k = 0
   and k huge, punctuation / case / hyphens / digits, greedy narrowing.
4. Round-trip: JSON, pickle and .pirz loads give byte-identical query results;
   δ+varbyte and δ+gamma decode to the exact original postings.
5. Compression codecs: variable-byte encodes 0/gappy/large gaps and round-trips;
   Elias-gamma round-trips and beats varbyte on size.
6. Tokenisation ablation (section 8): stopwords-removed phrase systems lose
   coverage (1 empty → 0... measured as the effect on derived-phrase recall);
   stemming is a flagged ablation only.
7. Paired bootstrap on per-query AP (10,000 resamples), seeded.
8. Biword false positives and positional filter equivalence (Level 3E).

All results above are from the actual run — rerun with
`python scripts/run_experiments.py` for a fresh table.

## 15. Design decisions and justifications

- **Phrase/proximity queries keep stopwords.**  Removing them both shifts
  positions and breaks multi-word phrases (`"king of spain"`, `"to be or not
  to be"`); it is a documented ablation only.  The query-time
  stopword-drop applies only to *content-word derivation* of queries
  (section 13), not to the index.
- **No stemming anywhere in the shipped system** (pure-IR boundary, section 1).
  Stemming appears only as a flagged ablation flag in the tokenizer
  (`TokenizerOptions(stem=True)`).
- **Positions are 1-based with "adjacent = distance 1"**, the convention that
  keeps NEAR/1 ↔ phrase symmetry; some systems (Google/Lucene "slop")
  count intervening words instead — that choice is documented throughout.
- **All metrics implemented from scratch** (~300 lines) rather than piling on
  dependencies; cross-checked against hand-computed examples.

## 16. Limitations

- Cranfield is tiny; the experiments measure correctness, not real-world
  scale.  Skip-pointers, varbyte compression and proximity scoring were
  measured on 1,400 docs, where the gains are modest.
- The nearest-doc machinery (`min_distance` / why-not) reports the *closest
  two terms* of a pair; for n > 2 the report is a minimal-covering-window
  diagnostic.
- Cranfield's qrels are noisy and lenient on "background" grades (raw 3).
  All reported numbers assume std grade ≥ 2; a sensitivity column at std ≥ 3
  appears in `results/tables.md`.
- The toy corpus is deliberately small (10 docs); it is the eyeball
  verification corpus and the functional test fixture.

## 17. Future work

- **Structural (sequential-dependence) weighting** of BM25+proximity
  (Metzler & Croft), with lambda tuned per query, e.g. via a small
  held-out set.
- **Compression tuning**: gap reduction, front coding, frame of reference;
  measure on a realistic 1M+ document collection.
- **Mined phrases (Level 3F) used as a phrase test set**, evaluated with the
  qrels mapped to their mined phrases, and a phrase-query variant of
  crossover with the Boolean AND.
- **Proximity-aware ranking (Level 3C) tuned** with λ learned per
  collection, plus a paired significance test over a bootstrapped sample.
- **Skip pointers interpolated** (:sqrt(n)-style, ~2·n ln n skips) and
  benchmarked on a larger collection.
- **Phrase index vs biword vs positional** extended to 4-5 word phrases with
  an empirical false-positive model.

## 18. References

- C. J. van Rijsbergen, *Information Retrieval*, 2nd ed., 1979 — the bible
  for inverted indexes, phrases and proximity (ch. 2.4, 5).
- C. J. van Rijsbergen, *The Geometry of Phrase Searches*, 1983.
- W. Büttcher, C. D. Moffat, M. W. van den Berg, *Efficient Phrase Query
  Processing*, 2006.
- W. Büttcher, C. D. Moffat, N. Satuluri, *Map-Reduce and the Fixed-Point
  Accumulation Framework*, 2006.
- R. Tao, C. Zhai, *Applying Proximity Weighting to Phrase Queries*, 2007.
- D. Metzler, W. Büttcher, C. Zhai, *Sequential Dependence Language Models
  for Information Retrieval*, 2007.
- W. Büttcher, C. D. Moffat, M. W. van den Berg, *Sustainable Retrieval:
  Toward Cheap and Fast Phrase Search via Word-Level Compression*, 2010.
- The Cranfield collection: J. L. Cleverdon (1967), author of the
  `cranqrel.readme` shipped in `data/cranfield/`; the standard reference
  numbers and the inverted relevance scale come from that README.
- M. Mitra, C. Cardie (1999, PhD thesis *High-Precision Information Retrieval*);
  M. Mitra, P. Majumder, FIRE organisers — the research profile behind this
  project's terminology; confirm the assignment rubric with the professor or a
  TA (dataset formats acceptable, Boolean-only vs ranked, ordered vs unordered
  proximity).
