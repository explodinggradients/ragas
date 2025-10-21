"""Utility functions for metrics migration E2E tests."""

from typing import Any, Dict, Optional

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import MetricResult


def create_legacy_sample(
    data: Dict[str, Any],
    user_input_key: str = "user_input",
    response_key: str = "response",
    reference_key: Optional[str] = "reference",
    retrieved_contexts_key: Optional[str] = "retrieved_contexts",
) -> SingleTurnSample:
    """Create a SingleTurnSample from a data dictionary for legacy metrics.

    Args:
        data: Dictionary containing sample data
        user_input_key: Key for user input in data dict
        response_key: Key for response in data dict
        reference_key: Key for reference in data dict (optional)
        retrieved_contexts_key: Key for retrieved contexts in data dict (optional)

    Returns:
        SingleTurnSample instance
    """
    kwargs = {
        "user_input": data.get(user_input_key, "dummy"),
    }

    if response_key and response_key in data:
        kwargs["response"] = data[response_key]

    if reference_key and reference_key in data:
        kwargs["reference"] = data[reference_key]

    if retrieved_contexts_key and retrieved_contexts_key in data:
        kwargs["retrieved_contexts"] = data[retrieved_contexts_key]

    return SingleTurnSample(**kwargs)


def compare_scores_with_tolerance(
    legacy_score: float,
    v2_score: float,
    tolerance: float,
    case_description: str,
    case_num: int,
) -> None:
    """Compare scores and assert they are within tolerance.

    Args:
        legacy_score: Score from legacy implementation
        v2_score: Score from v2 implementation
        tolerance: Maximum allowed difference
        case_description: Description of the test case
        case_num: Test case number

    Raises:
        AssertionError: If scores differ by more than tolerance
    """
    score_diff = abs(legacy_score - v2_score)
    assert score_diff < tolerance, (
        f"Case {case_num} ({case_description}): "
        f"Large difference: {legacy_score} vs {v2_score} (diff: {score_diff})"
    )


def assert_score_types(legacy_score: Any, v2_result: MetricResult) -> None:
    """Assert that scores have correct types and values are in valid range.

    Args:
        legacy_score: Score from legacy implementation
        v2_result: MetricResult from v2 implementation

    Raises:
        AssertionError: If types or ranges are invalid
    """
    assert isinstance(legacy_score, float), (
        f"Legacy score should be float, got {type(legacy_score)}"
    )
    assert isinstance(v2_result, MetricResult), (
        f"V2 result should be MetricResult, got {type(v2_result)}"
    )
    assert 0.0 <= legacy_score <= 1.0, f"Legacy score out of range: {legacy_score}"
    assert 0.0 <= v2_result.value <= 1.0, f"V2 score out of range: {v2_result.value}"


def print_test_header(
    metric_name: str,
    case_num: int,
    description: str,
    additional_info: Optional[Dict[str, str]] = None,
) -> None:
    """Print a standardized test case header.

    Args:
        metric_name: Name of the metric being tested
        case_num: Test case number
        description: Description of the test case
        additional_info: Optional dictionary of additional info to print
    """
    print(f"\n🧪 Testing {metric_name} - Case {case_num}: {description}")
    if additional_info:
        for key, value in additional_info.items():
            # Truncate long values
            display_value = value[:100] + "..." if len(value) > 100 else value
            print(f"   {key}: {display_value}")


def print_score_comparison(
    legacy_score: float,
    v2_score: float,
    precision: int = 6,
) -> None:
    """Print a standardized score comparison.

    Args:
        legacy_score: Score from legacy implementation
        v2_score: Score from v2 implementation
        precision: Number of decimal places to display
    """
    score_diff = abs(legacy_score - v2_score)
    print(f"   Legacy:    {legacy_score:.{precision}f}")
    print(f"   V2 Class:  {v2_score:.{precision}f}")
    print(f"   Diff:      {score_diff:.{precision}f}")


def print_test_success(message: str = "Scores within tolerance!") -> None:
    """Print a standardized success message.

    Args:
        message: Success message to display
    """
    print(f"   ✅ {message}")


def print_metric_specific_info(metric_name: str, description: str) -> None:
    """Print metric-specific test information.

    Args:
        metric_name: Name of the metric
        description: Description of what's being tested
    """
    print(f"\n🎯 Testing {metric_name}: {description}")
