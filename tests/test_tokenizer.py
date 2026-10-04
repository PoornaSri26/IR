"""Tokenizer unit tests (Level 2.2 edge cases)."""

from pir.tokenizer import (
    STOPWORDS,
    TokenizerOptions,
    _stem,
    tokenize,
    tokenize_with_spans,
)


def test_lowercase_split_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_hyphens_and_digits():
    assert tokenize("x-ray") == ["x", "ray"]
    assert tokenize("3.5 mm") == ["3", "5", "mm"]


def test_digits_kept_by_default():
    assert tokenize("version 2 and 10") == ["version", "2", "and", "10"]
    opts = TokenizerOptions(keep_digits=False)
    assert tokenize("version 2 and 10", opts) == ["version", "and"]


def test_stopwords_kept_by_default():
    assert "of" in tokenize("king of Spain")
    opts = TokenizerOptions(drop_stopwords=True)
    assert tokenize("king of Spain", opts) == ["king", "spain"]


def test_case_fold_off_ablation():
    opts = TokenizerOptions(case_fold=False)
    assert tokenize("Hello World", opts) == ["Hello", "World"]
    assert tokenize("Hello World") == ["hello", "world"]


def test_spans_match_terms():
    text = "The x-ray, a 3.5 mm plate."
    toks = tokenize(text)
    spans = tokenize_with_spans(text)
    assert [t for t, _, _ in spans] == toks
    assert text[spans[1][1] : spans[1][2]] == "x"
    assert text[spans[-1][1] : spans[-1][2]] == "plate"


def test_empty_and_none():
    assert tokenize("") == []
    assert tokenize("... !!!") == []
    assert tokenize(None) == []


def test_stopword_list_has_content():
    assert "the" in STOPWORDS and "of" in STOPWORDS


def test_toy_stemmer_flagged_behavior():
    opts = TokenizerOptions(stem=True)
    assert tokenize("indexes", opts) == ["index"]
    assert _stem("class") == "class"  # 'ss' intact
    # main system default: no stemming
    assert tokenize("indexes") == ["indexes"]
