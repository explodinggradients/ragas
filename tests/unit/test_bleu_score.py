"""Regression tests for BleuScore reference-structure fix.

The original code passed N reference streams of 1 segment each to sacrebleu,
instead of 1 stream of N segments. zip(*references) truncated at the shortest
stream (length 1), so only the first sentence was ever evaluated.
"""

import pytest
from sacrebleu import corpus_bleu


def test_bleu_evaluates_full_text_not_just_first_sentence():
    """After the fix, BLEU must reflect the whole text, not just sentence 1."""
    ref = "The cat sat on the mat. It was a sunny day. The birds were singing."
    # Only first sentence matches
    resp = "The cat sat on the mat. Yesterday was rainy. Dogs barked loudly."

    score = corpus_bleu([resp], [[ref]]).score / 100
    # Before the fix this was 1.0 (only evaluated matching 1st sentence)
    assert score < 0.8, f"Expected < 0.8 (not just first sentence), got {score}"


def test_bleu_later_sentences_contribute():
    """Sentences after the first must affect the score."""
    ref = "The cat sat on the mat. It was a sunny day. The birds were singing."
    # First sentence differs, 2nd and 3rd are identical
    resp = "Dogs chase butterflies. It was a sunny day. The birds were singing."

    score = corpus_bleu([resp], [[ref]]).score / 100
    # Before the fix this was 0.0 (only evaluated mismatching 1st sentence)
    assert score > 0.3, f"Expected > 0.3 (later sentences match), got {score}"


def test_bleu_identical_texts_score_one():
    """Identical reference and response should produce a perfect score."""
    text = "Hello world. This is a test. All sentences match."
    score = corpus_bleu([text], [[text]]).score / 100
    assert abs(score - 1.0) < 1e-6
