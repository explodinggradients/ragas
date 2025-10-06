"""Unit tests for V2BaseMetric."""

import pytest
from pydantic import ValidationError

from ragas.metrics.result import MetricResult
from ragas.metrics.v2.base import V2BaseMetric


class SimpleTestMetric(V2BaseMetric):
    """Simple test metric for unit testing."""

    name: str = "simple_test"
    multiplier: float = 1.0

    async def _ascore_impl(self, value: float) -> MetricResult:
        """Multiply the input value."""
        return MetricResult(value=value * self.multiplier)


class TestV2BaseMetric:
    """Test suite for V2BaseMetric."""

    def test_initialization(self):
        """Test basic metric initialization."""
        metric = SimpleTestMetric()
        assert metric.name == "simple_test"
        assert metric.allowed_values == (0.0, 1.0)
        assert metric.multiplier == 1.0

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        metric = SimpleTestMetric(multiplier=2.5, allowed_values=(0.0, 10.0))
        assert metric.multiplier == 2.5
        assert metric.allowed_values == (0.0, 10.0)

    def test_pydantic_validation_on_init(self):
        """Test that Pydantic validates on initialization."""
        # Note: Pydantic coerces lists to tuples, so we test with invalid values instead

        # Invalid allowed_values (min >= max)
        with pytest.raises(ValidationError):
            SimpleTestMetric(allowed_values=(1.0, 0.0))

    def test_pydantic_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            SimpleTestMetric(invalid_field="value")  # type: ignore

    @pytest.mark.asyncio
    async def test_ascore_basic(self):
        """Test async scoring."""
        metric = SimpleTestMetric(multiplier=2.0)
        result = await metric.ascore(value=0.5)

        assert result.value == 1.0
        assert isinstance(result, MetricResult)

    @pytest.mark.asyncio
    async def test_ascore_validation(self):
        """Test that results are validated against allowed_values."""
        metric = SimpleTestMetric(multiplier=10.0, allowed_values=(0.0, 1.0))
        result = await metric.ascore(value=0.5)

        # Result is 5.0, which exceeds allowed_values (0.0, 1.0)
        assert result.value is None
        assert result.reason is not None
        assert "expected value in range" in result.reason

    @pytest.mark.asyncio
    async def test_abatch_score(self):
        """Test batch scoring with concurrency."""
        metric = SimpleTestMetric(multiplier=2.0)
        inputs = [{"value": 0.1}, {"value": 0.2}, {"value": 0.3}]

        results = await metric.abatch_score(inputs)

        assert len(results) == 3
        assert results[0].value == 0.2
        assert results[1].value == 0.4
        assert results[2].value == 0.6

    def test_score_sync(self):
        """Test synchronous scoring."""
        metric = SimpleTestMetric(multiplier=3.0)
        result = metric.score(value=0.25)

        assert result.value == 0.75

    def test_score_from_async_context_raises(self):
        """Test that score() raises when called from async context."""
        import asyncio

        metric = SimpleTestMetric()

        async def try_sync_in_async():
            with pytest.raises(RuntimeError, match="Use ascore\\(\\) instead"):
                metric.score(value=0.5)

        asyncio.run(try_sync_in_async())

    def test_batch_score_sync(self):
        """Test synchronous batch scoring."""
        metric = SimpleTestMetric(multiplier=2.0)
        inputs = [{"value": 0.1}, {"value": 0.2}]

        results = metric.batch_score(inputs)

        assert len(results) == 2
        assert results[0].value == 0.2
        assert results[1].value == 0.4

    def test_save_config(self):
        """Test configuration serialization."""
        metric = SimpleTestMetric(multiplier=2.5, allowed_values=(0.0, 10.0))
        config = metric.save_config(exclude_components=False)

        assert config["name"] == "simple_test"
        assert config["multiplier"] == 2.5
        assert config["allowed_values"] == (0.0, 10.0)

    def test_model_dump(self):
        """Test Pydantic model_dump."""
        metric = SimpleTestMetric(multiplier=3.0)
        data = metric.model_dump()

        assert data["name"] == "simple_test"
        assert data["multiplier"] == 3.0
        assert data["allowed_values"] == (0.0, 1.0)

    def test_repr(self):
        """Test string representation."""
        metric = SimpleTestMetric(multiplier=2.0)
        repr_str = repr(metric)

        assert "SimpleTestMetric" in repr_str
        assert "name='simple_test'" in repr_str
        # Check that multiplier is present in repr
        assert "multiplier" in repr_str


class TestV2BaseMetricValidation:
    """Test validation logic."""

    def test_validate_result_value_valid(self):
        """Test validation of valid result values."""
        metric = SimpleTestMetric()

        assert metric._validate_result_value(0.0) is None
        assert metric._validate_result_value(0.5) is None
        assert metric._validate_result_value(1.0) is None

    def test_validate_result_value_out_of_range(self):
        """Test validation of out-of-range values."""
        metric = SimpleTestMetric()

        error = metric._validate_result_value(1.5)
        assert error is not None
        assert "expected value in range" in error

        error = metric._validate_result_value(-0.1)
        assert error is not None
        assert "expected value in range" in error

    def test_validate_result_value_non_numeric(self):
        """Test validation of non-numeric values."""
        metric = SimpleTestMetric()

        error = metric._validate_result_value("invalid")
        assert error is not None
        assert "expected a numeric value" in error

        error = metric._validate_result_value(None)
        assert error is not None
        assert "expected a numeric value" in error


class TestV2BaseMetricFieldValidators:
    """Test Pydantic field validators."""

    def test_allowed_values_validator_invalid_type(self):
        """Test allowed_values validator with invalid types."""
        # Wrong tuple length should raise validation error
        with pytest.raises(ValidationError):
            SimpleTestMetric(allowed_values=(0.0,))  # type: ignore

    def test_allowed_values_validator_non_numeric(self):
        """Test allowed_values validator with non-numeric values."""
        # Pydantic tries to coerce strings to numbers, so use truly invalid types
        with pytest.raises(ValidationError):
            SimpleTestMetric(allowed_values=("invalid", "values"))  # type: ignore

    def test_allowed_values_validator_inverted_range(self):
        """Test allowed_values validator with min >= max."""
        with pytest.raises(ValidationError, match="must be less than max"):
            SimpleTestMetric(allowed_values=(1.0, 1.0))

        with pytest.raises(ValidationError, match="must be less than max"):
            SimpleTestMetric(allowed_values=(2.0, 1.0))
