"""Unit tests for ContextEntityRecall's recall computation.

``_compute_recall_score`` is a pure function of its arguments (it does not use
``self``), so we can exercise it directly without constructing the LLM-backed
metric.
"""

from ragas.metrics.collections import ContextEntityRecall

_recall = ContextEntityRecall._compute_recall_score


def test_perfect_recall_is_exactly_one():
    # Every reference entity is present in the context -> recall must be 1.0.
    # Previously the "+ 1e-8" epsilon made this 0.99999999.
    score = _recall(None, ["paris", "france"], ["paris", "france", "eiffel"])
    assert score == 1.0


def test_partial_recall():
    # 1 of 2 reference entities present.
    assert _recall(None, ["paris", "france"], ["paris"]) == 0.5


def test_duplicate_entities_are_deduplicated():
    # Sets dedupe, so repeats don't inflate the denominator or numerator.
    assert _recall(None, ["paris", "paris"], ["paris"]) == 1.0


def test_no_reference_entities_returns_zero():
    # Nothing to recall; must not raise a ZeroDivisionError.
    assert _recall(None, [], ["paris"]) == 0.0
