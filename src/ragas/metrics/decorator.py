"""decorator factory for creating custom metrics"""

__all__ = ["create_metric_decorator"]

import asyncio
import inspect
import typing as t
import warnings
from dataclasses import dataclass, field
from typing import get_args, get_origin, get_type_hints

from pydantic import ValidationError, create_model

from .result import MetricResult


def create_metric_decorator(metric_class):
    """
    Factory function that creates decorator factories for different metric types.

    Args:
        metric_class: The metric class to use (DiscreteMetrics, NumericMetrics, etc.)

    Returns:
        A decorator factory function for the specified metric type
    """

    def decorator_factory(
        name: t.Optional[str] = None,
        **metric_params,
    ):
        """
        Creates a decorator that wraps a function into a metric instance.

        Args:
            name: Optional name for the metric (defaults to function name)
            **metric_params: Additional parameters specific to the metric type
                (values for DiscreteMetrics, range for NumericMetrics, etc.)

        Returns:
            A decorator function
        """

        def decorator(func):
            # Get metric name and check if function is async
            metric_name = name or func.__name__
            is_async = inspect.iscoroutinefunction(func)
            sig = inspect.signature(func)

            # TODO: Move to dataclass type implementation
            @dataclass
            class CustomMetric(metric_class):
                _func: t.Any = field(default=None, init=False)

                def _validate_result_value(self, result_value):
                    """Validate result value based on metric type constraints."""
                    # Discrete metric validation
                    if hasattr(self, "allowed_values") and isinstance(
                        self.allowed_values, list
                    ):
                        if result_value not in self.allowed_values:
                            return f"Metric {self.name} returned '{result_value}' but expected one of {self.allowed_values}"

                    # Numeric metric validation
                    if hasattr(self, "allowed_values") and isinstance(
                        self.allowed_values, (tuple, range)
                    ):
                        if not isinstance(result_value, (int, float)):
                            return f"Metric {self.name} returned '{result_value}' but expected a numeric value"

                        if isinstance(self.allowed_values, tuple):
                            min_val, max_val = self.allowed_values
                            if not (min_val <= result_value <= max_val):
                                return f"Metric {self.name} returned {result_value} but expected value in range {self.allowed_values}"
                        elif isinstance(self.allowed_values, range):
                            if result_value not in self.allowed_values:
                                return f"Metric {self.name} returned {result_value} but expected value in range {self.allowed_values}"

                    # Ranking metric validation
                    if hasattr(self, "allowed_values") and isinstance(
                        self.allowed_values, int
                    ):
                        if not isinstance(result_value, list):
                            return f"Metric {self.name} returned '{result_value}' but expected a list"
                        if len(result_value) != self.allowed_values:
                            return f"Metric {self.name} returned list of length {len(result_value)} but expected {self.allowed_values} items"

                    return None  # No validation error

                def _create_positional_error(self, args: tuple, kwargs: dict) -> str:
                    """Create error message for positional arguments."""
                    func_param_names = list(sig.parameters.keys())

                    msg = f"\n❌ {self.name}.score() requires keyword arguments, not positional.\n\n"
                    msg += (
                        f"   You provided: score({', '.join(repr(a) for a in args)})\n"
                    )
                    msg += "   Correct usage: score("

                    corrections = []
                    for i, param_name in enumerate(func_param_names):
                        if i < len(args):
                            corrections.append(f"{param_name}={repr(args[i])}")
                        else:
                            corrections.append(f"{param_name}=...")

                    msg += ", ".join(corrections) + ")\n\n"
                    msg += "   💡 Tip: Always use parameter names for clarity and future compatibility."

                    return msg

                def _create_pydantic_model(self):
                    """Create a Pydantic model dynamically from the function signature."""
                    try:
                        type_hints = get_type_hints(func)
                    except (NameError, AttributeError):
                        type_hints = {}

                    field_definitions = {}

                    for name, param in sig.parameters.items():
                        # Get type hint, default to str if no hint available
                        type_hint = type_hints.get(name, param.annotation)
                        if type_hint == inspect.Parameter.empty:
                            type_hint = str

                        # Get default value
                        if param.default != inspect.Parameter.empty:
                            default = param.default
                        else:
                            # Check if it's an optional type
                            origin = get_origin(type_hint)
                            if origin is t.Union and type(None) in get_args(type_hint):
                                # Optional type, default to None
                                default = None
                            else:
                                # Required field
                                default = ...

                        field_definitions[name] = (type_hint, default)

                    # Create the dynamic model
                    model_name = f"{self.name}_ValidationModel"
                    return create_model(model_name, **field_definitions)

                def _format_pydantic_errors(
                    self, validation_error: ValidationError
                ) -> str:
                    """Format Pydantic validation errors into user-friendly messages."""
                    msg = f"\n❌ Type validation errors for {self.name}:\n\n"

                    for error in validation_error.errors():
                        field = error["loc"][0]
                        error_msg = error["msg"]
                        input_value = error.get("input", "N/A")

                        msg += f"   - {field}: {error_msg} (got: {repr(input_value)})\n"

                    return msg

                def _validate_inputs(self, args: tuple, kwargs: dict):
                    """Validate all inputs using Pydantic with helpful error messages."""
                    # Check for positional arguments (keep custom helpful error)
                    if args:
                        raise TypeError(self._create_positional_error(args, kwargs))

                    # Create dynamic Pydantic model from function signature
                    try:
                        pydantic_model = self._create_pydantic_model()
                    except Exception as e:
                        # Fallback if model creation fails
                        warnings.warn(
                            f"Could not create validation model: {e}", UserWarning
                        )
                        return

                    # Warn about unknown arguments (but continue processing)
                    valid_params = set(pydantic_model.model_fields.keys())
                    unknown = set(kwargs.keys()) - valid_params

                    if unknown:
                        warnings.warn(
                            f"⚠️  {self.name} received unknown arguments: {', '.join(sorted(unknown))}\n"
                            f"   Valid arguments: {', '.join(sorted(valid_params))}",
                            UserWarning,
                        )

                    # Validate using Pydantic (only for valid parameters)
                    valid_kwargs = {
                        k: v for k, v in kwargs.items() if k in valid_params
                    }

                    try:
                        # Pydantic handles missing required fields and type validation
                        validated_data = pydantic_model(**valid_kwargs)
                        # Store the validated data for use in execution
                        self._validated_data = validated_data.model_dump()
                    except ValidationError as e:
                        raise TypeError(self._format_pydantic_errors(e))

                def score(self, *args, **kwargs):
                    """Synchronous scoring method that wraps ascore()."""

                    # Use asyncio.run to execute the async method
                    async def _async_wrapper():
                        return await self.ascore(*args, **kwargs)

                    # Check if we're already in an event loop
                    try:
                        # If we're in a running event loop, we need nest_asyncio for compatibility
                        _ = asyncio.get_running_loop()
                        # Import nest_asyncio style runner from ragas
                        from ragas.async_utils import run

                        return run(_async_wrapper())
                    except RuntimeError:
                        # No running event loop, safe to use asyncio.run
                        return asyncio.run(_async_wrapper())

                async def ascore(self, *args, **kwargs):
                    """Asynchronous scoring method."""
                    # Validate inputs before execution
                    self._validate_inputs(args, kwargs)

                    try:
                        # Use validated data from Pydantic if available
                        func_kwargs = getattr(self, "_validated_data", {})

                        # Execute the function based on its type
                        if is_async:
                            # For async functions, await the result
                            result = await func(**func_kwargs)
                        else:
                            # For sync functions, run directly
                            result = func(**func_kwargs)

                        # Ensure result is a MetricResult
                        if not isinstance(result, MetricResult):
                            # Wrap plain values in MetricResult
                            result = MetricResult(value=result, reason=None)

                        # Validate the result based on metric type
                        validation_error = self._validate_result_value(result.value)
                        if validation_error:
                            return MetricResult(value=None, reason=validation_error)

                        return result

                    except Exception as e:
                        # Handle errors gracefully
                        error_msg = f"Error executing metric {self.name}: {str(e)}"
                        return MetricResult(value=None, reason=error_msg)

                def __call__(self, *args, **kwargs):
                    """Make the metric instance directly callable using the original function."""
                    if self._func is None:
                        raise RuntimeError(
                            "Original function not set on metric instance"
                        )

                    if is_async:
                        # For async functions, always return the coroutine
                        # Let the caller handle async context appropriately
                        return self._func(*args, **kwargs)
                    else:
                        # For sync functions, just call directly
                        return self._func(*args, **kwargs)

            # Create the metric instance with all parameters
            metric_instance = CustomMetric(name=metric_name, **metric_params)

            # Store the original function for direct calling
            metric_instance._func = func

            # Preserve metadata
            metric_instance.__name__ = metric_name
            metric_instance.__doc__ = func.__doc__

            return metric_instance

        return decorator

    return decorator_factory
