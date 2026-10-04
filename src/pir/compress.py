"""Posting-list compression (Level 3A): delta encoding + variable-byte.

Method (Manning et al. ch. 5; Zobel & Moffat 2006):
- Within a posting list, doc ids are stored as GAPS between consecutive ids;
  positions are stored as gaps between consecutive positions, resetting the
  gap base at each new doc id (first position of a doc is a gap from 0).
- Every gap is encoded with variable bytes: 7 payload bits per byte, the high
  bit set on the LAST byte of a number.  Small gaps dominate real postings,
  so most numbers fit in one byte.

Elias-gamma is included as a second coder (used by tests; selectable for the
experiment in scripts/run_experiments.py).  Gamma represents n >= 1 as
(offset in binary of (n-1)) with a unary length prefix.

File layout used by index.save_compressed: see index.py.
"""

from __future__ import annotations

# --------------------------------------------------------------- variable byte


def vb_encode_number(n: int) -> bytes:
    """Encode one non-negative integer as variable bytes."""
    if n < 0:
        raise ValueError("variable-byte encoder needs non-negative integers")
    out = bytearray()
    out.append(n & 0x7F)
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)  # continuation byte (more to come)
        n >>= 7
    out.reverse()
    return bytes(out)


def vb_encode(nums) -> bytes:
    return b"".join(vb_encode_number(n) for n in nums)


def vb_decode(buf: bytes, pos: int = 0) -> tuple[list[int], int]:
    """Decode all numbers in buf[pos:]; returns (numbers, new_pos)."""
    out: list[int] = []
    i = pos
    while i < len(buf):
        n, i = vb_decode_one(buf, i)
        out.append(n)
    return out, i


def vb_decode_one(buf: bytes, pos: int) -> tuple[int, int]:
    """Decode ONE variable-byte number starting at pos; returns (n, new_pos)."""
    n = 0
    i = pos
    while True:
        b = buf[i]
        i += 1
        if b & 0x80:  # continuation: accumulate 7 bits
            n = (n << 7) | (b & 0x7F)
        else:  # last byte of a number
            n = (n << 7) | b
            return n, i


# ------------------------------------------------------------------ elias gamma


def gamma_encode(nums) -> bytes:
    """Encode a whole list as a single gamma-coded bit stream.

    Standard Elias-gamma for n >= 1: write b = bit_length(n) - 1 in unary
    ('1'*b + '0'), then n in binary without its leading 1 (b bits).
    Uses integer bitwise ops instead of string manipulation.
    """
    if not nums:
        return b""
    acc = 0    # accumulator (grows left)
    nbits = 0  # bits currently in accumulator
    out = bytearray()

    def _flush_bytes():
        nonlocal acc, nbits
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)

    for n in nums:
        if n < 1:
            raise ValueError("gamma coder represents integers >= 1")
        b = n.bit_length() - 1  # unary length
        # write b ones then one zero (b+1 bits total)
        if b:
            acc = (acc << b) | ((1 << b) - 1)  # b ones
            nbits += b
        acc = (acc << 1)  # the terminating zero
        nbits += 1
        _flush_bytes()
        # write the b least-significant bits of n (without its leading 1)
        if b:
            acc = (acc << b) | (n & ((1 << b) - 1))
            nbits += b
            _flush_bytes()

    # flush remaining bits (pad with zeros on the right)
    if nbits:
        acc <<= (8 - nbits)
        out.append(acc & 0xFF)
    return bytes(out)


def gamma_decode(buf: bytes, count: int) -> list[int]:
    """Decode exactly `count` gamma-coded numbers from buf.
    Uses integer bitwise ops instead of building a character string.
    """
    out: list[int] = []
    # Unpack all bytes into a single large integer (MSB first)
    total_bits = len(buf) * 8
    bits = int.from_bytes(buf, "big") if buf else 0
    pos = 0  # current bit position from the left (MSB = position 0)

    def _bit(p: int) -> int:
        return (bits >> (total_bits - 1 - p)) & 1

    for _ in range(count):
        b = 0
        while pos < total_bits and _bit(pos) == 1:
            b += 1
            pos += 1
        pos += 1  # skip the terminating '0'
        if b:
            rest = 0
            for _ in range(b):
                rest = (rest << 1) | _bit(pos)
                pos += 1
            out.append((1 << b) | rest)
        else:
            out.append(1)
    return out


# --------------------------------------------------------------------- delta


def delta_encode(sorted_nums: list[int]) -> list[int]:
    """Gaps of an ascending list; first element kept as-is (gap from 0).

    Works for any non-negative list (gaps may be 0 if the input has repeats).
    """
    out: list[int] = []
    prev = 0
    for n in sorted_nums:
        if n < prev:
            raise ValueError("delta_encode expects ascending input")
        out.append(n - prev)
        prev = n
    return out


def delta_decode(gaps: list[int]) -> list[int]:
    out: list[int] = []
    prev = 0
    for g in gaps:
        prev += g
        out.append(prev)
    return out


# --------------------------------------------------------- whole postings dict


def encode_postings_varbyte(postings: dict[str, dict[int, list[int]]]) -> bytes:
    """term -> {doc -> positions}  -->  one varbyte blob.

    Layout per term:
        len(term) vb | term bytes | ndocs vb | per doc: docgap vb | tf vb |
                    first pos vb | position gaps vb...
    Terms are written in sorted order so decoding is deterministic.
    """
    out = bytearray()
    for term in sorted(postings):
        tb = term.encode("utf-8")
        out += vb_encode_number(len(tb))
        out += tb
        docs = postings[term]
        out += vb_encode_number(len(docs))
        prev_doc = 0
        for doc in sorted(docs):
            out += vb_encode_number(doc - prev_doc)
            prev_doc = doc
            pos = docs[doc]
            out += vb_encode_number(len(pos))
            out += vb_encode(delta_encode(pos))
    return bytes(out)


def decode_postings_varbyte(blob: bytes) -> dict[str, dict[int, list[int]]]:
    pos = 0
    postings: dict[str, dict[int, list[int]]] = {}
    while pos < len(blob):
        tlen, pos = vb_decode_one(blob, pos)
        term = blob[pos : pos + tlen].decode("utf-8")
        pos += tlen
        ndocs, pos = vb_decode_one(blob, pos)
        docs: dict[int, list[int]] = {}
        prev_doc = 0
        for _ in range(ndocs):
            docgap, pos = vb_decode_one(blob, pos)
            doc = prev_doc + docgap
            prev_doc = doc
            tf, pos = vb_decode_one(blob, pos)
            gaps: list[int] = []
            last = 0
            for _ in range(tf):
                g, pos = vb_decode_one(blob, pos)
                last += g
                gaps.append(last)
            docs[doc] = gaps
        postings[term] = docs
    return postings


def encode_postings_gamma(postings: dict[str, dict[int, list[int]]]) -> bytes:
    """Same layout as varbyte, but ALL numbers go into ONE continuous gamma
    bit stream (as n+1, since gamma cannot represent 0).  Encoding the whole
    stream at once is essential: padding each number to a byte boundary would
    both waste space and break the decoder.
    """
    out = bytearray()
    nums: list[int] = []
    for term in sorted(postings):
        tb = term.encode("utf-8")
        out += vb_encode_number(len(tb))
        out += tb
        docs = postings[term]
        nums.append(len(docs) + 1)
        prev_doc = 0
        for doc in sorted(docs):
            nums.append(doc - prev_doc + 1)
            prev_doc = doc
            pos = docs[doc]
            nums.append(len(pos) + 1)
            nums.extend(g + 1 for g in delta_encode(pos))
    out += gamma_encode(nums)
    return bytes(out)


def decode_postings_gamma(
    blob: bytes, shape: dict[str, dict[int, list[int]]]
) -> dict[str, dict[int, list[int]]]:
    """Decode a gamma-coded postings blob.

    The gamma stream is self-delimiting per number but NOT per posting, so the
    caller supplies `shape` -- the same term/doc/tf skeleton as the original
    postings -- purely to know how many numbers to read.  Every gap VALUE is
    recovered from the stream itself, so this is a faithful decode, not a
    lookup of the answer.
    """
    header_len = 0
    for term in sorted(shape):
        tb = term.encode("utf-8")
        header_len += len(vb_encode_number(len(tb))) + len(tb)

    total = 0
    for term in sorted(shape):
        docs = shape[term]
        total += 1
        for doc in docs:
            total += 2 + len(docs[doc])

    nums = gamma_decode(blob[header_len:], total)
    it = iter(nums)
    out: dict[str, dict[int, list[int]]] = {}
    for term in sorted(shape):
        next(it)  # number of documents (informational)
        docs: dict[int, list[int]] = {}
        for doc in sorted(shape[term]):
            next(it)  # doc-id gap
            tf = next(it) - 1  # encoder stored (tf + 1), same 0-offset as gaps
            last = 0
            ps: list[int] = []
            for _ in range(tf):
                # encoder stored (gap + 1) because gamma cannot represent 0
                last += next(it) - 1
                ps.append(last)
            docs[doc] = ps
        out[term] = docs
    return out
