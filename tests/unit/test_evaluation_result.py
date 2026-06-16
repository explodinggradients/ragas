import math
import typing as t

import numpy as np
import pytest

from ragas.dataset_schema import EvaluationDataset, EvaluationResult


def make_result(scores: t.List[t.Dict[str, t.Any]]) -> EvaluationResult:
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": f"question {idx}",
                "response": f"answer {idx}",
                "reference": f"reference {idx}",
            }
            for idx in range(len(scores))
        ]
    )
    return EvaluationResult(scores=scores, dataset=dataset)


def test_summary_returns_aggregate_metric_scores():
    result = make_result(
        [
            {"faithfulness": 0.8, "context_recall": 0.9},
            {"faithfulness": 1.0, "context_recall": 0.7},
        ]
    )

    assert result.summary() == {
        "faithfulness": pytest.approx(0.9),
        "context_recall": pytest.approx(0.8),
    }


def test_summary_returns_a_copy():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 1.0}])

    summary = result.summary()
    summary["faithfulness"] = 0.0

    assert result.summary()["faithfulness"] == pytest.approx(0.9)


def test_check_thresholds_passes_min_score():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 0.9}])

    report = result.check_thresholds(min_scores={"faithfulness": 0.8})

    assert report["passed"] is True
    assert report["failures"] == []
    assert report["checks"]["faithfulness"]["passed"] is True
    assert report["checks"]["faithfulness"]["constraints"] == {"min": 0.8}


def test_check_thresholds_fails_min_score():
    result = make_result([{"faithfulness": 0.7}, {"faithfulness": 0.8}])

    report = result.check_thresholds(min_scores={"faithfulness": 0.8})

    assert report["passed"] is False
    assert report["checks"]["faithfulness"]["passed"] is False
    assert "faithfulness" in report["failures"][0]
    assert "0.75" in report["failures"][0]
    assert "minimum 0.8" in report["failures"][0]


def test_check_thresholds_supports_max_score():
    passing_result = make_result([{"error_rate": 0.05}, {"error_rate": 0.15}])
    failing_result = make_result([{"error_rate": 0.25}, {"error_rate": 0.35}])

    passing_report = passing_result.check_thresholds(max_scores={"error_rate": 0.2})
    failing_report = failing_result.check_thresholds(max_scores={"error_rate": 0.2})

    assert passing_report["passed"] is True
    assert failing_report["passed"] is False
    assert "maximum 0.2" in failing_report["failures"][0]


def test_check_thresholds_supports_ranges():
    passing_result = make_result([{"answer_relevancy": 0.8}, {"answer_relevancy": 0.9}])
    failing_result = make_result([{"answer_relevancy": 0.6}, {"answer_relevancy": 0.8}])

    passing_report = passing_result.check_thresholds(
        ranges={"answer_relevancy": (0.8, 1.0)}
    )
    failing_report = failing_result.check_thresholds(
        ranges={"answer_relevancy": (0.8, 1.0)}
    )

    assert passing_report["passed"] is True
    assert failing_report["passed"] is False
    assert "between 0.8 and 1.0" in failing_report["failures"][0]


def test_check_thresholds_fails_missing_metric_clearly():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 0.9}])

    report = result.check_thresholds(min_scores={"context_recall": 0.9})

    assert report["passed"] is False
    assert report["checks"]["context_recall"]["score"] is None
    assert "context_recall" in report["failures"][0]
    assert "missing from summary" in report["failures"][0]


def test_check_thresholds_fails_nan_score_clearly():
    result = make_result([{"faithfulness": np.nan}, {"faithfulness": np.nan}])

    report = result.check_thresholds(min_scores={"faithfulness": 0.8})

    assert report["passed"] is False
    assert math.isnan(report["checks"]["faithfulness"]["score"])
    assert "not a finite numeric score" in report["failures"][0]


def test_check_thresholds_fails_non_numeric_score_clearly():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 1.0}])
    result._repr_dict["faithfulness"] = "high"

    report = result.check_thresholds(min_scores={"faithfulness": 0.8})

    assert report["passed"] is False
    assert report["checks"]["faithfulness"]["score"] == "high"
    assert "not a finite numeric score" in report["failures"][0]


def test_check_thresholds_empty_thresholds_pass():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 1.0}])

    report = result.check_thresholds()

    assert report == {
        "passed": True,
        "scores": {"faithfulness": pytest.approx(0.9)},
        "checks": {},
        "failures": [],
    }


def test_check_thresholds_invalid_range_raises_value_error():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 1.0}])

    with pytest.raises(ValueError, match="minimum cannot exceed maximum"):
        result.check_thresholds(ranges={"faithfulness": (0.9, 0.8)})


def test_check_thresholds_invalid_threshold_value_raises_value_error():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 1.0}])

    with pytest.raises(ValueError, match="must be numeric"):
        result.check_thresholds(min_scores={"faithfulness": "high"})  # type: ignore[dict-item]


def test_check_thresholds_non_finite_threshold_value_raises_value_error():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 1.0}])

    with pytest.raises(ValueError, match="finite numeric"):
        result.check_thresholds(min_scores={"faithfulness": math.nan})


def test_assert_thresholds_passes_silently():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 0.9}])

    assert result.assert_thresholds(min_scores={"faithfulness": 0.8}) is None


def test_assert_thresholds_raises_assertion_error_on_failure():
    result = make_result([{"faithfulness": 0.7}, {"faithfulness": 0.8}])

    with pytest.raises(AssertionError) as exc_info:
        result.assert_thresholds(min_scores={"faithfulness": 0.8})

    error_message = str(exc_info.value)
    assert "faithfulness" in error_message
    assert "0.75" in error_message
    assert "constraints" in error_message
    assert "minimum 0.8" in error_message


def test_check_thresholds_relative_passes():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 0.9}])

    report = result.check_thresholds(
        relative_thresholds={"faithfulness": 0.1},
        baseline_scores={"faithfulness": 0.85},
    )

    assert report["passed"] is True
    assert "faithfulness" in report["checks"]
    assert report["checks"]["faithfulness"]["passed"] is True
    assert report["checks"]["faithfulness"]["constraints"]["relative"] == 0.1
    assert report["checks"]["faithfulness"]["constraints"]["baseline"] == 0.85


def test_check_thresholds_relative_fails():
    result = make_result([{"faithfulness": 0.5}, {"faithfulness": 0.6}])

    report = result.check_thresholds(
        relative_thresholds={"faithfulness": 0.05},
        baseline_scores={"faithfulness": 0.85},
    )

    assert report["passed"] is False
    assert report["checks"]["faithfulness"]["passed"] is False
    assert "outside relative threshold" in report["failures"][0]


def test_check_thresholds_relative_requires_baseline():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 0.9}])

    with pytest.raises(ValueError, match="baseline_scores"):
        result.check_thresholds(relative_thresholds={"faithfulness": 0.1})


def test_check_thresholds_relative_requires_matching_baseline():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 0.9}])

    with pytest.raises(ValueError, match="missing metric"):
        result.check_thresholds(
            relative_thresholds={"faithfulness": 0.1},
            baseline_scores={"other_metric": 0.5},
        )


def test_check_thresholds_relative_must_be_between_zero_and_one():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 0.9}])

    with pytest.raises(ValueError, match="between 0 and 1"):
        result.check_thresholds(
            relative_thresholds={"faithfulness": 1.5},
            baseline_scores={"faithfulness": 0.85},
        )


def test_assert_thresholds_relative_silent():
    result = make_result([{"faithfulness": 0.8}, {"faithfulness": 0.9}])

    assert result.assert_thresholds(
        relative_thresholds={"faithfulness": 0.1},
        baseline_scores={"faithfulness": 0.85},
    ) is None


def test_assert_thresholds_relative_raises():
    result = make_result([{"faithfulness": 0.5}, {"faithfulness": 0.6}])

    with pytest.raises(AssertionError, match="relative threshold"):
        result.assert_thresholds(
            relative_thresholds={"faithfulness": 0.05},
            baseline_scores={"faithfulness": 0.85},
        )


def test_check_thresholds_combined_absolute_and_relative():
    result = make_result(
        [
            {"faithfulness": 0.8, "answer_relevancy": 0.9},
            {"faithfulness": 0.9, "answer_relevancy": 0.92},
        ]
    )

    report = result.check_thresholds(
        min_scores={"faithfulness": 0.8},
        relative_thresholds={"answer_relevancy": 0.05},
        baseline_scores={"answer_relevancy": 0.91},
    )

    assert report["passed"] is True
    assert report["checks"]["faithfulness"]["passed"] is True
    assert report["checks"]["answer_relevancy"]["passed"] is True


def test_existing_getitem_and_to_pandas_behavior_remains_intact():
    result = make_result(
        [
            {"faithfulness": 0.8, "context_recall": 0.9},
            {"faithfulness": 1.0, "context_recall": 0.7},
        ]
    )

    assert result["faithfulness"] == [0.8, 1.0]

    df = result.to_pandas()
    assert df["user_input"].tolist() == ["question 0", "question 1"]
    assert df["faithfulness"].tolist() == [0.8, 1.0]
