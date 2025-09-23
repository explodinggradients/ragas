"""Tests for metric decorators (discrete_metric, numeric_metric, ranking_metric)

This module tests that the decorators can handle both:
1. Functions returning plain values (strings, floats, lists)
2. Functions returning MetricResult objects

Following TDD approach: Write failing tests first, then implement the fix.
"""

import typing as t

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


class TestErrorHandling:
    """Test comprehensive error handling and validation."""

    def test_positional_arguments_error(self):
        """Test that positional arguments give helpful error message."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        with pytest.raises(TypeError) as exc_info:
            my_metric.score("test", "test")

        error_msg = str(exc_info.value)
        assert "requires keyword arguments, not positional" in error_msg
        assert "You provided: score('test', 'test')" in error_msg
        assert "Correct usage: score(predicted='test', expected='test')" in error_msg
        assert "💡 Tip:" in error_msg

    def test_missing_required_arguments_error(self):
        """Test error message for missing required arguments."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str, context: str) -> str:
            return "pass"

        with pytest.raises(TypeError) as exc_info:
            my_metric.score(predicted="test")

        error_msg = str(exc_info.value)
        assert "Missing required arguments" in error_msg
        assert "expected" in error_msg
        assert "context" in error_msg
        assert "Example:" in error_msg

    def test_missing_required_arguments_with_optional_arguments_error(self):
        """Test that Optional[T] parameters are treated as optional, not required."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(
            predicted: str, expected: str, context: t.Optional[str] = None
        ) -> str:
            return "pass"

        with pytest.raises(TypeError) as exc_info:
            my_metric.score(
                predicted="test"
            )  # missing 'expected' but 'context' is optional

        error_msg = str(exc_info.value)
        assert "Missing required arguments" in error_msg
        assert "expected" in error_msg
        assert "context" not in error_msg  # context should not be listed as required
        assert "Example:" in error_msg

    def test_optional_type_annotation_without_default(self):
        """Test that t.Optional[T] without default value is still treated as optional."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str, context: t.Optional[str]) -> str:
            return "pass"

        # Should work without the optional parameter
        result = my_metric.score(predicted="test", expected="test")
        assert result.value == "pass"

        # Should also work with the optional parameter
        result = my_metric.score(
            predicted="test", expected="test", context="some context"
        )
        assert result.value == "pass"

        # Should also work with None for the optional parameter
        result = my_metric.score(predicted="test", expected="test", context=None)
        assert result.value == "pass"

    def test_mixed_required_optional_and_default_parameters(self):
        """Test complex scenario with required, optional, and default parameters."""

        @discrete_metric(name="complex_metric", allowed_values=["pass", "fail"])
        def my_metric(
            required1: str,
            required2: int,
            optional_typed: t.Optional[str],  # Optional type annotation
            with_default: float = 0.5,  # Has default value
            optional_with_default: t.Optional[
                str
            ] = None,  # Both optional and has default
        ) -> str:
            return "pass"

        # Test missing required arguments
        with pytest.raises(TypeError) as exc_info:
            my_metric.score(required1="test")  # missing required2

        error_msg = str(exc_info.value)
        assert "Missing required arguments" in error_msg
        assert "required2" in error_msg
        assert "optional_typed" not in error_msg  # Should not be required
        assert "with_default" not in error_msg  # Should not be required
        assert "optional_with_default" not in error_msg  # Should not be required

        # Test that it works with just required arguments
        result = my_metric.score(required1="test", required2=42)
        assert result.value == "pass"

        # Test that it works with all arguments
        result = my_metric.score(
            required1="test",
            required2=42,
            optional_typed="optional",
            with_default=0.8,
            optional_with_default="also optional",
        )
        assert result.value == "pass"

    def test_type_validation_error(self):
        """Test type validation with helpful error messages."""

        @numeric_metric(name="score", allowed_values=(0, 1))
        def my_metric(predicted: float, expected: int) -> float:
            return float(predicted - expected)

        with pytest.raises(TypeError) as exc_info:
            my_metric.score(predicted="not_a_float", expected="not_an_int")

        error_msg = str(exc_info.value)
        assert "Type mismatch" in error_msg
        assert "predicted: expected float" in error_msg
        assert "expected: expected int" in error_msg

    def test_unknown_arguments_warning(self):
        """Test that unknown arguments generate warnings."""

        @discrete_metric(name="simple", allowed_values=["pass", "fail"])
        def my_metric(text: str) -> str:
            return "pass"

        with pytest.warns(UserWarning, match="received unknown arguments"):
            result = my_metric.score(text="test", unknown_param="value")

        # Should still work despite unknown parameter
        assert result.value == "pass"

    def test_mixed_error_scenarios(self):
        """Test combinations of errors."""

        @discrete_metric(name="complex", allowed_values=["pass", "fail"])
        def my_metric(text: str, threshold: float = 0.5) -> str:
            return "pass"

        # Test positional + extra args
        with pytest.raises(TypeError, match="requires keyword arguments"):
            my_metric.score("text", 0.5, extra="unknown")

    def test_optional_parameters_work(self):
        """Test that optional parameters don't cause missing args error."""

        @discrete_metric(name="optional_test", allowed_values=["pass", "fail"])
        def my_metric(text: str, threshold: float = 0.5) -> str:
            return "pass" if len(text) > threshold else "fail"

        # Should work with just required parameter
        result = my_metric.score(text="hello")
        assert result.value == "pass"

        # Should also work with optional parameter
        result = my_metric.score(text="hi", threshold=5.0)
        assert result.value == "fail"

    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test that async methods also validate inputs."""

        @discrete_metric(name="async_metric", allowed_values=["pass", "fail"])
        async def my_metric(text: str) -> str:
            return "pass"

        # Test positional args error in async
        with pytest.raises(TypeError, match="requires keyword arguments"):
            await my_metric.ascore("test")

        # Test missing args error in async
        with pytest.raises(TypeError, match="Missing required arguments"):
            await my_metric.ascore()

    def test_helpful_examples_in_errors(self):
        """Test that error messages include helpful examples based on type hints."""

        @numeric_metric(name="complex_metric", allowed_values=(0, 10))
        def my_metric(score: int, weight: float, tags: list) -> float:
            return float(score * weight)

        with pytest.raises(TypeError) as exc_info:
            my_metric.score()  # Missing all args

        error_msg = str(exc_info.value)
        # Should have examples based on type hints
        assert "score=1" in error_msg  # int example
        assert "weight=0.5" in error_msg  # float example
        assert 'tags=["item1", "item2"]' in error_msg  # list example

    def test_no_type_hints_still_works(self):
        """Test that metrics work even without type hints."""

        @discrete_metric(name="no_hints", allowed_values=["pass", "fail"])
        def my_metric(text, threshold=0.5):  # No type hints
            return "pass"

        # Should still validate and work
        result = my_metric.score(text="hello")
        assert result.value == "pass"

        # Should still catch positional args
        with pytest.raises(TypeError, match="requires keyword arguments"):
            my_metric.score("hello", 0.8)

    def test_type_validation_shows_full_type_for_generics(self):
        """Test that type error messages show full types for both simple and generic types."""

        @discrete_metric(name="complex_types", allowed_values=["pass", "fail"])
        def my_metric(
            simple_str: str,
            simple_int: int,
            optional_str: t.Optional[str] = None,
            list_of_strings: t.List[str] = None,
            union_type: t.Union[str, int] = "default",
        ) -> str:
            return "pass"

        # Test 1: Simple types should show correctly (baseline)
        with pytest.raises(TypeError) as exc_info:
            my_metric.score(
                simple_str=123, simple_int="not_int"
            )  # Wrong types for simple params

        error_msg = str(exc_info.value)
        assert "simple_str: expected str, got int" in error_msg
        assert "simple_int: expected int, got str" in error_msg

        # Test 2: Optional types should show "Optional[T]", not just "Optional"
        with pytest.raises(TypeError) as exc_info:
            my_metric.score(
                simple_str="ok", simple_int=1, optional_str=[1, 2, 3]
            )  # list instead of Optional[str]

        error_msg = str(exc_info.value)
        assert "optional_str: expected Optional[str], got list" in error_msg
        assert (
            "optional_str: expected Optional, got list" not in error_msg
        )  # This should NOT appear

        # Test 3: Union types should show "Union[T1, T2]", not just "Union"
        with pytest.raises(TypeError) as exc_info:
            my_metric.score(
                simple_str="ok", simple_int=1, union_type=[1, 2, 3]
            )  # list instead of Union[str, int]

        error_msg = str(exc_info.value)
        assert "union_type: expected Union[str, int], got list" in error_msg
        assert (
            "union_type: expected Union, got list" not in error_msg
        )  # This should NOT appear

        # Note: List[str] validation is not currently implemented in _check_type method
        # So we skip that test for now - the focus is on fixing the error message formatting
