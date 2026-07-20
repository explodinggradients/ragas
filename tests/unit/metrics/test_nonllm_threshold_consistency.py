"""Tests for NonLLMContextRecall and NonLLMContextPrecisionWithReference threshold consistency.

These metrics share a `threshold` parameter (default 0.5) and use the same
NonLLMStringSimilarity distance measure. They must apply the threshold consistently
to avoid a context matching exactly at the boundary being counted as relevant by one
metric and irrelevant by the other.
"""

from ragas.metrics._context_recall import NonLLMContextRecall
from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference


def test_boundary_threshold_equivalence():
    """Both metrics must count a score exactly at the threshold as relevant."""
    boundary_score = 0.5

    recall_result = NonLLMContextRecall(threshold=0.5)._compute_score(
        [boundary_score]
    )
    precision_result = NonLLMContextPrecisionWithReference(
        threshold=0.5
    )._calculate_average_precision([1])

    assert recall_result == 1.0, (
        f"Recall should count threshold-match as relevant, got {recall_result}"
    )

    assert precision_result == 1.0, (
        f"Precision should count threshold-match as relevant, got {precision_result}"
    )


def test_all_scores_below_threshold():
    """Scores below threshold should yield zero."""
    scores = [0.49, 0.0, 0.3]

    recall_result = NonLLMContextRecall(threshold=0.5)._compute_score(scores)
    assert recall_result == 0.0, (
        f"Expected 0.0 for all-scores-below-threshold, got {recall_result}"
    )


def test_all_scores_above_threshold():
    """Scores above threshold should yield 1.0 for recall."""
    scores = [0.51, 0.99, 0.6]

    recall_result = NonLLMContextRecall(threshold=0.5)._compute_score(scores)
    assert recall_result == 1.0, (
        f"Expected 1.0 for all-scores-above-threshold, got {recall_result}"
    )


def test_mixed_scores():
    """Mixed above/below threshold scores should compute proportion correctly."""
    scores = [0.9, 0.4, 0.6, 0.3]

    recall_result = NonLLMContextRecall(threshold=0.5)._compute_score(scores)
    assert recall_result == 0.5, (
        f"Expected 0.5 (2/4 relevant), got {recall_result}"
    )


def test_custom_threshold():
    """Both metrics respect custom threshold values."""
    scores = [0.8, 0.75, 0.9, 0.7]

    recall_result = NonLLMContextRecall(threshold=0.75)._compute_score(scores)
    # With >= 0.75: [1, 1, 1, 0] -> 3/4
    assert recall_result == 0.75, (
        f"Expected 0.75 (3/4) at threshold 0.75, got {recall_result}"
    )
