#!/usr/bin/env python
"""Download the Cranfield test collection into data/cranfield/.

Cranfield = 1,400 aeronautics abstracts, 225 queries, graded relevance
judgements.  It is the classic small IR collection with qrels, which is what
makes P@k / MAP / nDCG evaluation possible.

Sources are tried in order (the Glasgow host has been unreliable at times,
so mirrors are included).  Everything lands in data/cranfield/:
    cran.all.1400   documents (fields .I .T .A .B .W)
    cran.qry        225 queries (.I id / .W text)
    cranqrel        relevance judgements (qid docid rel, or qid iter docid rel)

Usage:  python scripts/download_cranfield.py
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "cranfield")

TARBALLS = [
    "https://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz",
    "http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz",
]

# Individual-file mirrors (repo/file triplets), used if no tarball works.
MIRRORS = [
    (
        "https://raw.githubusercontent.com/castorini/anserini-data/master/Cranfield",
        ["cran.all.1400", "cran.qry", "cranqrel"],
    ),
    (
        "https://raw.githubusercontent.com/castorini/anserini-data/main/Cranfield",
        ["cran.all.1400", "cran.qry", "cranqrel"],
    ),
    (
        "https://raw.githubusercontent.com/ijabz/trec-cranfield-dataset/master",
        ["cran.all.1400", "cran.qry", "cranqrel"],
    ),
]

UA = {"User-Agent": "Mozilla/5.0 (positional-ir course project)"}


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def try_tarballs() -> bool:
    for url in TARBALLS:
        print(f"  trying {url}")
        try:
            blob = fetch(url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"    failed: {e}")
            continue
        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                for m in members:
                    base = os.path.basename(m.name)
                    if base in ("cran.all.1400", "cran.qry", "cranqrel"):
                        data = tf.extractfile(m).read()
                        with open(os.path.join(DEST, base), "wb") as f:
                            f.write(data)
                        print(f"    extracted {base} ({len(data)} bytes)")
                return True
        except tarfile.TarError as e:
            print(f"    not a valid tar.gz: {e}")
    return False


def try_mirrors() -> bool:
    for base, files in MIRRORS:
        print(f"  trying mirror {base}")
        got = []
        for name in files:
            try:
                data = fetch(f"{base}/{name}")
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                print(f"    {name}: failed ({e})")
                continue
            with open(os.path.join(DEST, name), "wb") as f:
                f.write(data)
            print(f"    {name}: {len(data)} bytes")
            got.append(name)
        if len(got) == len(files):
            return True
    return False


def verify() -> bool:
    need = ["cran.all.1400", "cran.qry", "cranqrel"]
    missing = [n for n in need if not os.path.exists(os.path.join(DEST, n))]
    if missing:
        print(f"MISSING: {missing}")
        return False
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from pir.datasets import load_cranfield

    docs, queries, qrels = load_cranfield(DEST)
    print(
        f"OK: {len(docs)} documents, {len(queries)} queries, "
        f"{sum(len(v) for v in qrels.values())} judgements over {len(qrels)} queries"
    )
    if len(docs) < 1000 or len(queries) < 100:
        print("WARNING: collection looks smaller than expected (1400 docs / 225 queries)")
        return False
    sample = sorted(queries, key=int)[:1]
    if sample:
        print(f"  first query ({sample[0]}): {queries[sample[0]][:90]}...")
    return True


def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    print("Downloading the Cranfield collection into", DEST)
    ok = try_tarballs() or try_mirrors()
    if not ok:
        print(
            "\nAutomatic download failed. Fetch the files manually into",
            DEST,
            "from any of these and re-run:\n"
            "  https://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz\n"
            "  https://github.com/castorini/anserini-data (directory: Cranfield)\n"
            "  https://huggingface.co/datasets (search: cranfield)\n"
            "Needed: cran.all.1400, cran.qry, cranqrel\n"
            "Any mirror of the SAME files works — the collection is public domain-ish\n"
            "(Cleverdon 1967; often redistributed freely for teaching).",
            file=sys.stderr,
        )
        return 1
    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())