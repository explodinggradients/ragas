"""Tests for metric decorators (discrete_metric, numeric_metric, ranking_metric)

This module tests that the decorators can handle both:
1. Functions returning plain values (strings, floats, lists)
2. Functions returning MetricResult objects

Following TDD approach: Write failing tests first, then implement the fix.
"""

import pytest

from ragas.metrics import MetricResult, discrete_metric, numeric_metric, ranking_metric


class TestDiscreteMetric:
    """Tests for discrete_metric decorator."""

    def test_discrete_metric_with_plain_string_return(self):
        """Test discrete metric with function returning plain string."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        # This should work without errors
        result = my_metric.score(predicted="test", expected="test")

        assert isinstance(result, MetricResult)
        assert result.value == "pass"
        assert result.reason is None  # Should be None for plain value returns

    def test_discrete_metric_with_plain_string_fail(self):
        """Test discrete metric returning 'fail'."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        result = my_metric.score(predicted="hello", expected="world")

        assert isinstance(result, MetricResult)
        assert result.value == "fail"
        assert result.reason is None

    def test_discrete_metric_with_metric_result_return(self):
        """Test discrete metric with function returning MetricResult."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> MetricResult:
            value = "pass" if predicted.lower() == expected.lower() else "fail"
            reason = f"Compared '{predicted}' with '{expected}'"
            return MetricResult(value=value, reason=reason)

        result = my_metric.score(predicted="test", expected="test")

        assert isinstance(result, MetricResult)
        assert result.value == "pass"
        assert result.reason == "Compared 'test' with 'test'"

    def test_discrete_metric_validation_invalid_value(self):
        """Test discrete metric validation with invalid value."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "maybe"  # Invalid value

        result = my_metric.score(predicted="test", expected="test")

        assert isinstance(result, MetricResult)
        assert result.value is None
        assert "expected one of ['pass', 'fail']" in result.reason

    @pytest.mark.asyncio
    async def test_discrete_metric_async_with_plain_return(self):
        """Test async discrete metric with plain string return."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        async def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        result = await my_metric.ascore(predicted="test", expected="test")

        assert isinstance(result, MetricResult)
        assert result.value == "pass"
        assert result.reason is None


class TestNumericMetric:
    """Tests for numeric_metric decorator."""

    def test_numeric_metric_with_plain_float_return(self):
        """Test numeric metric with function returning plain float."""

        @numeric_metric(name="response_accuracy", allowed_values=(0, 1))
        def my_metric(predicted: float, expected: float) -> float:
            return abs(predicted - expected) / max(expected, 1e-5)

        result = my_metric.score(predicted=0.8, expected=1.0)

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert abs(result.value - 0.2) < 1e-10
        assert result.reason is None

    def test_numeric_metric_with_metric_result_return(self):
        """Test numeric metric with function returning MetricResult."""

        @numeric_metric(name="response_accuracy", allowed_values=(0, 1))
        def my_metric(predicted: float, expected: float) -> MetricResult:
            value = abs(predicted - expected) / max(expected, 1e-5)
            reason = f"Difference: {abs(predicted - expected)}"
            return MetricResult(value=value, reason=reason)

        result = my_metric.score(predicted=0.8, expected=1.0)

        assert isinstance(result, MetricResult)
        assert abs(result.value - 0.2) < 1e-10
        assert result.reason == "Difference: 0.19999999999999996"

    def test_numeric_metric_validation_out_of_range(self):
        """Test numeric metric validation with out-of-range value."""

        @numeric_metric(name="response_accuracy", allowed_values=(0, 1))
        def my_metric(predicted: float, expected: float) -> float:
            return 1.5  # Out of range

        result = my_metric.score(predicted=0.8, expected=1.0)

        assert isinstance(result, MetricResult)
        assert result.value is None
        assert "expected value in range (0, 1)" in result.reason

    @pytest.mark.asyncio
    async def test_numeric_metric_async_with_plain_return(self):
        """Test async numeric metric with plain float return."""

        @numeric_metric(name="response_accuracy", allowed_values=(0, 1))
        async def my_metric(predicted: float, expected: float) -> float:
            return abs(predicted - expected) / max(expected, 1e-5)

        result = await my_metric.ascore(predicted=0.8, expected=1.0)

        assert isinstance(result, MetricResult)
        assert abs(result.value - 0.2) < 1e-10
        assert result.reason is None


class TestRankingMetric:
    """Tests for ranking_metric decorator."""

    def test_ranking_metric_with_plain_list_return(self):
        """Test ranking metric with function returning plain list."""

        @ranking_metric(name="response_ranking", allowed_values=3)
        def my_metric(responses: list) -> list:
            response_lengths = [len(response) for response in responses]
            sorted_indices = sorted(
                range(len(response_lengths)), key=lambda i: response_lengths[i]
            )
            return sorted_indices

        result = my_metric.score(
            responses=["short", "a bit longer", "the longest response"]
        )

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, list)
        assert result.value == [0, 1, 2]  # indices sorted by length
        assert result.reason is None

    def test_ranking_metric_with_metric_result_return(self):
        """Test ranking metric with function returning MetricResult."""

        @ranking_metric(name="response_ranking", allowed_values=3)
        def my_metric(responses: list) -> MetricResult:
            response_lengths = [len(response) for response in responses]
            sorted_indices = sorted(
                range(len(response_lengths)), key=lambda i: response_lengths[i]
            )
            reason = f"Sorted by lengths: {response_lengths}"
            return MetricResult(value=sorted_indices, reason=reason)

        result = my_metric.score(
            responses=["short", "a bit longer", "the longest response"]
        )

        assert isinstance(result, MetricResult)
        assert result.value == [0, 1, 2]
        assert result.reason == "Sorted by lengths: [5, 12, 20]"

    def test_ranking_metric_validation_wrong_length(self):
        """Test ranking metric validation with wrong list length."""

        @ranking_metric(name="response_ranking", allowed_values=3)
        def my_metric(responses: list) -> list:
            return [0, 1]  # Wrong length - should be 3

        result = my_metric.score(responses=["short", "medium", "long"])

        assert isinstance(result, MetricResult)
        assert result.value is None
        assert "expected 3 items" in result.reason

    @pytest.mark.asyncio
    async def test_ranking_metric_async_with_plain_return(self):
        """Test async ranking metric with plain list return."""

        @ranking_metric(name="response_ranking", allowed_values=2)
        async def my_metric(responses: list) -> list:
            return [1, 0]  # Reverse order

        result = await my_metric.ascore(responses=["first", "second"])

        assert isinstance(result, MetricResult)
        assert result.value == [1, 0]
        assert result.reason is None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_discrete_metric_with_custom_allowed_values(self):
        """Test discrete metric with custom allowed values."""

        @discrete_metric(
            name="sentiment", allowed_values=["positive", "negative", "neutral"]
        )
        def sentiment_metric(text: str) -> str:
            if "good" in text.lower():
                return "positive"
            elif "bad" in text.lower():
                return "negative"
            else:
                return "neutral"

        result = sentiment_metric.score(text="This is good")
        assert result.value == "positive"

        result = sentiment_metric.score(text="This is bad")
        assert result.value == "negative"

        result = sentiment_metric.score(text="This is okay")
        assert result.value == "neutral"

    def test_numeric_metric_with_range_type(self):
        """Test numeric metric with range type."""

        @numeric_metric(name="score", allowed_values=range(0, 11))  # 0-10
        def score_metric(value: int) -> int:
            return min(10, max(0, value))

        result = score_metric.score(value=5)
        assert result.value == 5

        result = score_metric.score(value=15)  # Should be clamped to 10
        assert result.value == 10

    def test_function_with_no_parameters(self):
        """Test metric function with no parameters."""

        @discrete_metric(name="constant", allowed_values=["always_pass"])
        def constant_metric() -> str:
            return "always_pass"

        result = constant_metric.score()
        assert result.value == "always_pass"

    def test_function_with_exception(self):
        """Test that exceptions are handled gracefully."""

        @discrete_metric(name="error_metric", allowed_values=["pass", "fail"])
        def error_metric(should_error: bool) -> str:
            if should_error:
                raise ValueError("Test error")
            return "pass"

        # Should not raise exception, should return error result
        result = error_metric.score(should_error=True)

        assert isinstance(result, MetricResult)
        assert result.value is None
        assert "Error executing metric" in result.reason
        assert "Test error" in result.reason
