"""Test suite for validating metric score tolerance."""

import logging
import os
import typing as t

import pytest

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
)
from tests.e2e.test_dataset_utils import load_amnesty_dataset_safe

if t.TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

# Loading the dataset
amnesty_qa = load_amnesty_dataset_safe("english_v3")  # type: ignore


def assert_in_range(score: float, value: float, plus_or_minus: float):
    """
    Check if computed score is within the range of value +/- plus_or_minus.
    """
    assert value - plus_or_minus <= score <= value + plus_or_minus


# Acceptable score tolerances for each metric
# Format: (metric_name, metric, expected_value, tolerance)
METRIC_TOLERANCES = [
    ("answer_relevancy", answer_relevancy, 0.85, 0.20),
    ("faithfulness", faithfulness, 0.75, 0.20),
    ("context_recall", context_recall, 0.80, 0.20),
    ("context_precision", context_precision, 0.80, 0.20),
    ("answer_correctness", answer_correctness, 0.65, 0.25),
    ("answer_similarity", answer_similarity, 0.75, 0.25),
    ("context_entity_recall", context_entity_recall, 0.75, 0.20),
]


@pytest.mark.ragas_ci
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.parametrize(
    "metric_name,metric,expected_value,tolerance",
    METRIC_TOLERANCES,
    ids=[m[0] for m in METRIC_TOLERANCES],
)
def test_metric_tolerance(metric_name, metric, expected_value, tolerance):
    """
    Test that each metric produces scores within expected tolerance.

    This test validates that metrics produce consistent scores within acceptable
    tolerance ranges when evaluated on the amnesty dataset.
    """
    # Evaluate the metric on a single sample
    result = evaluate(
        EvaluationDataset.from_hf_dataset(t.cast("Dataset", amnesty_qa))[:1],
        metrics=[metric],
        show_progress=False,
    )

    assert result is not None, f"Evaluation result is None for {metric_name}"
    assert isinstance(result, EvaluationResult), "Expected EvaluationResult"

    # Get the score for the metric
    score = result[metric_name][0]

    # Calculate tolerance range
    min_val = expected_value - tolerance
    max_val = expected_value + tolerance
    passed = min_val <= score <= max_val
    status = "✓" if passed else "✗"

    # Print to stdout (always visible)
    print(
        f"\n{metric_name:.<30} Score: {score:.4f} | Tolerance: [{min_val:.2f}, {max_val:.2f}] | {status} {'PASSED' if passed else 'FAILED'}"
    )

    # Also log for --log-cli-level=INFO
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing metric: {metric_name}")
    logger.info(f"Expected tolerance: {min_val:.2f} - {max_val:.2f}")
    logger.info(f"Score: {score:.4f}")
    logger.info(f"Status: {status} {'PASSED' if passed else 'FAILED'}")
    logger.info(f"{'=' * 60}\n")

    # Assert the score is within tolerance
    assert_in_range(score, value=expected_value, plus_or_minus=tolerance)


@pytest.mark.ragas_ci
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_all_metrics_together():
    """
    Test all metrics together to validate overall evaluation functionality.

    This test runs all metrics in a single evaluation and validates that:
    1. All metrics complete successfully
    2. All scores are within expected tolerance
    """
    all_metrics = [metric for _, metric, _, _ in METRIC_TOLERANCES]

    result = evaluate(
        EvaluationDataset.from_hf_dataset(t.cast("Dataset", amnesty_qa))[:1],
        metrics=all_metrics,
        show_progress=False,
    )

    assert result is not None, "Evaluation result is None"
    assert isinstance(result, EvaluationResult), "Expected EvaluationResult"

    # Validate each metric's score
    results_summary = []
    all_passed = True

    for metric_name, _, expected_value, tolerance in METRIC_TOLERANCES:
        score = result[metric_name][0]
        min_val = expected_value - tolerance
        max_val = expected_value + tolerance
        passed = min_val <= score <= max_val
        status = "✓" if passed else "✗"

        results_summary.append(
            {
                "metric": metric_name,
                "score": score,
                "expected_tolerance": f"[{min_val:.2f}, {max_val:.2f}]",
                "status": status,
                "passed": passed,
            }
        )

        if not passed:
            all_passed = False

    # Print summary table to stdout (always visible)
    print("\n" + "=" * 90)
    print("METRIC TOLERANCE SUMMARY - ALL METRICS TOGETHER")
    print("=" * 90)
    print(f"{'Metric':<30} {'Score':<12} {'Tolerance Range':<20} {'Status':<10}")
    print("-" * 90)

    for res in results_summary:
        print(
            f"{res['metric']:<30} {res['score']:<12.4f} {res['expected_tolerance']:<20} {res['status']:<10}"
        )

    print("-" * 90)
    print(f"Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    print("=" * 90 + "\n")

    # Also log for --log-cli-level=INFO
    logger.info("\n" + "=" * 80)
    logger.info("Testing all metrics together")
    logger.info("=" * 80)
    logger.info("\nMetric Tolerance Summary:")
    logger.info("-" * 80)
    logger.info(
        f"{'Metric':<25} {'Score':<10} {'Expected Tolerance':<20} {'Status':<10}"
    )
    logger.info("-" * 80)

    for res in results_summary:
        logger.info(
            f"{res['metric']:<25} {res['score']:<10.4f} {res['expected_tolerance']:<20} {res['status']:<10}"
        )

    logger.info("-" * 80)
    logger.info(f"\nOverall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    logger.info("=" * 80 + "\n")

    # Assert all metrics passed
    assert all_passed, "Some metrics failed to meet expected tolerance"


@pytest.mark.ragas_ci
def test_assert_in_range():
    """Test the assert_in_range helper function."""
    assert_in_range(0.51, value=0.5, plus_or_minus=0.1)
    assert_in_range(0.5, value=0.5, plus_or_minus=0.0)
    assert_in_range(0.75, value=0.8, plus_or_minus=0.1)
