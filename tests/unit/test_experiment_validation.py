"""Tests for experiment parameter validation functionality."""

import pytest

from ragas.experimental import Dataset, experiment


class TestExperimentValidation:
    """Test cases for experiment parameter validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.dataset = Dataset(name="test_dataset", backend="inmemory")
        self.dataset.append({"question": "What is 2+2?", "context": "Math"})
        self.dataset.append({"question": "What is 3+3?", "context": "More math"})

    @pytest.mark.asyncio
    async def test_valid_single_parameter_experiment(self):
        """Test experiment with correct single parameter."""

        @experiment()
        async def single_param_experiment(row):
            return {"result": f"Answer to: {row['question']}", "score": 1.0}

        # Should work without errors
        result = await single_param_experiment.arun(self.dataset)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_valid_multi_parameter_experiment(self):
        """Test experiment with multiple parameters provided correctly."""

        @experiment()
        async def multi_param_experiment(row, evaluator_llm, flag=True):
            return {
                "result": f"Answer using {evaluator_llm}",
                "flag": flag,
                "score": 1.0,
            }

        # Should work when all required parameters are provided
        result = await multi_param_experiment.arun(
            self.dataset, evaluator_llm="gpt-4", flag=False
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_missing_required_parameter(self):
        """Test that missing required parameters raise ValueError."""

        @experiment()
        async def multi_param_experiment(row, evaluator_llm, flag=True):
            return {"result": "test", "score": 1.0}

        # Should raise ValueError when required parameter is missing
        with pytest.raises(ValueError) as exc_info:
            await multi_param_experiment.arun(self.dataset, abc=123)

        error_msg = str(exc_info.value)
        assert "Parameter validation failed" in error_msg
        assert "multi_param_experiment()" in error_msg
        assert "evaluator_llm (required)" in error_msg
        assert "missing a required argument: 'evaluator_llm'" in error_msg

    @pytest.mark.asyncio
    async def test_validation_catches_parameter_binding_errors(self):
        """Test that validation catches various parameter binding issues."""

        @experiment()
        async def strict_param_experiment(row, required_str, required_int=10):
            return {"result": "test", "score": 1.0}

        # Test 1: Wrong keyword argument name should fail validation
        with pytest.raises(ValueError) as exc_info:
            await strict_param_experiment.arun(
                self.dataset,
                wrong_param_name="value",
                name="test",  # This should fail
            )

        error_msg = str(exc_info.value)
        assert "Parameter validation failed" in error_msg
        assert "strict_param_experiment()" in error_msg

        # Test 2: Valid call should work
        result = await strict_param_experiment.arun(
            self.dataset, required_str="valid_value", required_int=20, name="test"
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_unexpected_keyword_arguments(self):
        """Test that unexpected keyword arguments raise ValueError."""

        @experiment()
        async def single_param_experiment(row):
            return {"result": "test", "score": 1.0}

        # Should raise ValueError when unexpected keyword argument
        with pytest.raises(ValueError) as exc_info:
            await single_param_experiment.arun(self.dataset, unexpected_kwarg="value")

        error_msg = str(exc_info.value)
        assert "Parameter validation failed" in error_msg
        assert "single_param_experiment()" in error_msg

    @pytest.mark.asyncio
    async def test_validation_happens_before_backend_resolution(self):
        """Test that validation occurs before any backend setup."""

        @experiment()
        async def invalid_experiment(row, required_param):
            return {"result": "test", "score": 1.0}

        # Should fail immediately without trying to resolve invalid backend
        with pytest.raises(ValueError) as exc_info:
            await invalid_experiment.arun(
                self.dataset,
                backend="nonexistent_backend",  # This would normally fail later
            )

        # Should get validation error, not backend resolution error
        error_msg = str(exc_info.value)
        assert "Parameter validation failed" in error_msg
        assert "required_param (required)" in error_msg

    @pytest.mark.asyncio
    async def test_empty_dataset_skips_validation(self):
        """Test that empty datasets skip validation."""

        @experiment()
        async def invalid_experiment(row, required_param):
            return {"result": "test", "score": 1.0}

        empty_dataset = Dataset(name="empty", backend="inmemory")

        # Should not raise validation error for empty dataset
        result = await invalid_experiment.arun(empty_dataset)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_function_with_kwargs(self):
        """Test experiment function that accepts **kwargs."""

        @experiment()
        async def kwargs_experiment(row, **kwargs):
            return {"result": f"Row: {row['question']}", "kwargs": kwargs, "score": 1.0}

        # Should work with additional keyword arguments
        result = await kwargs_experiment.arun(
            self.dataset, extra_param="value", another_param=42
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_function_with_args_and_kwargs(self):
        """Test experiment function with *args and **kwargs."""

        @experiment()
        async def flexible_experiment(row, *args, **kwargs):
            return {
                "result": f"Row: {row['question']}",
                "args": args,
                "kwargs": kwargs,
                "score": 1.0,
            }

        # Should work with additional keyword arguments that get passed to **kwargs
        result = await flexible_experiment.arun(
            self.dataset,
            name="test",
            extra_kwarg="value",  # This will be passed to **kwargs
            another_kwarg=42,  # This will also be passed to **kwargs
        )
        assert len(result) == 2
