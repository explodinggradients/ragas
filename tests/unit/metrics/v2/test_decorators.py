"""Unit tests for v2 metric decorators."""

import pytest

from ragas.metrics.result import MetricResult
from ragas.metrics.v2.base import V2BaseMetric
from ragas.metrics.v2.decorators import v2_metric, v2_numeric_metric


class TestV2MetricDecorator:
    """Test suite for v2_metric decorator."""

    def test_decorator_creates_class_instance(self):
        """Test that decorator creates a V2BaseMetric instance."""

        @v2_metric(name="test_metric")
        async def my_metric(value: float) -> MetricResult:
            return MetricResult(value=value)

        # Should be an instance of V2BaseMetric
        assert isinstance(my_metric, V2BaseMetric)
        assert my_metric.name == "test_metric"

    def test_decorator_default_name(self):
        """Test that decorator uses function name as default."""

        @v2_metric()
        async def custom_metric_name(value: float) -> MetricResult:
            return MetricResult(value=value)

        assert custom_metric_name.name == "custom_metric_name"

    def test_decorator_custom_allowed_values(self):
        """Test decorator with custom allowed_values."""

        @v2_metric(name="score", allowed_values=(0.0, 10.0))
        async def my_metric(value: float) -> MetricResult:
            return MetricResult(value=value)

        assert my_metric.allowed_values == (0.0, 10.0)

    @pytest.mark.asyncio
    async def test_decorator_async_function(self):
        """Test decorator with async function."""

        @v2_metric(name="async_test")
        async def async_metric(value: float) -> MetricResult:
            return MetricResult(value=value * 2.0, reason="Doubled")

        result = await async_metric.ascore(value=5.0)

        assert result.value == 10.0
        assert result.reason == "Doubled"

    def test_decorator_sync_function(self):
        """Test decorator with sync function."""

        @v2_metric(name="sync_test")
        def sync_metric(value: float) -> MetricResult:
            return MetricResult(value=value * 3.0)

        result = sync_metric.score(value=2.0)

        assert result.value == 6.0

    @pytest.mark.asyncio
    async def test_decorator_auto_wraps_non_metricresult(self):
        """Test that decorator wraps non-MetricResult returns."""

        @v2_metric(name="auto_wrap")
        async def simple_metric(value: float) -> float:
            return value * 2.0

        result = await simple_metric.ascore(value=3.0)

        assert isinstance(result, MetricResult)
        assert result.value == 6.0

    @pytest.mark.asyncio
    async def test_decorator_batch_scoring(self):
        """Test batch scoring with decorated metric."""

        @v2_metric(name="batch_test")
        async def metric(value: float) -> float:
            return value * 2.0

        results = await metric.abatch_score(
            [
                {"value": 1.0},
                {"value": 2.0},
                {"value": 3.0},
            ]
        )

        assert len(results) == 3
        assert results[0].value == 2.0
        assert results[1].value == 4.0
        assert results[2].value == 6.0

    def test_decorator_serialization(self):
        """Test that decorated metrics can be serialized."""

        @v2_metric(name="serializable", allowed_values=(0.0, 100.0))
        async def metric(value: float) -> float:
            return value

        config = metric.save_config()

        assert config["name"] == "serializable"
        assert config["allowed_values"] == (0.0, 100.0)

    def test_decorator_with_custom_parameters(self):
        """Test decorator with custom metric parameters."""

        @v2_metric(name="custom_params", threshold=5.0, description="Test metric")
        async def metric(value: float, threshold: float = 5.0) -> float:
            return min(value, threshold)

        # Custom parameters should be accessible
        assert hasattr(metric, "threshold")
        assert metric.threshold == 5.0

    def test_decorator_preserves_docstring(self):
        """Test that decorator preserves function docstring."""

        @v2_metric(name="documented")
        async def metric(value: float) -> float:
            """This is a test metric with documentation."""
            return value

        assert metric.__doc__ == "This is a test metric with documentation."

    @pytest.mark.asyncio
    async def test_decorator_validation(self):
        """Test that decorated metrics validate results."""

        @v2_metric(name="validated", allowed_values=(0.0, 1.0))
        async def metric(value: float) -> float:
            return value

        # Valid result
        result = await metric.ascore(value=0.5)
        assert result.value == 0.5

        # Invalid result (out of range)
        result = await metric.ascore(value=2.0)
        assert result.value is None
        assert result.reason is not None
        assert "expected value in range" in result.reason


class TestV2NumericMetricDecorator:
    """Test suite for v2_numeric_metric decorator."""

    def test_numeric_metric_decorator(self):
        """Test v2_numeric_metric decorator."""

        @v2_numeric_metric(name="numeric", allowed_values=(0.0, 10.0))
        async def metric(value: float) -> float:
            return value

        assert isinstance(metric, V2BaseMetric)
        assert metric.name == "numeric"
        assert metric.allowed_values == (0.0, 10.0)

    @pytest.mark.asyncio
    async def test_numeric_metric_scoring(self):
        """Test numeric metric scoring."""

        @v2_numeric_metric(name="score", allowed_values=(0.0, 100.0))
        async def percentage_metric(correct: int, total: int) -> float:
            return (correct / total) * 100 if total > 0 else 0.0

        result = await percentage_metric.ascore(correct=85, total=100)

        assert result.value == 85.0


class TestDecoratorEdgeCases:
    """Test edge cases for decorators."""

    @pytest.mark.asyncio
    async def test_decorator_with_exception(self):
        """Test decorator when function raises exception."""

        @v2_metric(name="error_metric")
        async def failing_metric(value: float) -> float:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await failing_metric.ascore(value=1.0)

    def test_decorator_callable_backward_compat(self):
        """Test that decorated metric can be called like a function."""

        @v2_metric(name="callable_test")
        def metric(value: float) -> float:
            return value * 2.0

        # Should be callable like the original function
        # (This is for backward compatibility)
        assert callable(metric)
