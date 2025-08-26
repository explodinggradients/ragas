"""decorator factory for creating custom metrics"""

__all__ = ["create_metric_decorator"]

import asyncio
import inspect
import typing as t
from dataclasses import dataclass

from ragas.llms import InstructorBaseRagasLLM as BaseRagasLLM

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
            llm: The language model instance to use
            prompt: The prompt template
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

            # Check function signature to determine if it expects llm/prompt
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            expects_llm = "llm" in param_names
            expects_prompt = "prompt" in param_names

            # TODO: Move to dataclass type implementation
            @dataclass
            class CustomMetric(metric_class):
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

                def _run_sync_in_async(self, func, *args, **kwargs):
                    """Run a synchronous function in an async context."""
                    # For sync functions, just run them normally
                    return func(*args, **kwargs)

                def _execute_metric(self, llm, is_async_execution, **kwargs):
                    """Execute the metric function with proper async handling."""
                    try:
                        # Prepare function arguments based on what the function expects
                        func_kwargs = kwargs.copy()
                        func_args = []

                        if expects_llm:
                            func_args.append(llm)
                        if expects_prompt:
                            func_args.append(self.prompt)

                        if is_async:
                            # Async function implementation
                            if is_async_execution:
                                # In async context, await the function directly
                                result = func(*func_args, **func_kwargs)
                            else:
                                # In sync context, run the async function in an event loop
                                try:
                                    loop = asyncio.get_event_loop()
                                except RuntimeError:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                result = loop.run_until_complete(
                                    func(*func_args, **func_kwargs)
                                )
                        else:
                            # Sync function implementation
                            result = func(*func_args, **func_kwargs)

                        # Ensure result is a MetricResult
                        if not isinstance(result, MetricResult):
                            raise ValueError(
                                f"Custom metric function must return MetricResult, got {type(result)}"
                            )

                        # Validate the result based on metric type
                        validation_error = self._validate_result_value(result.value)
                        if validation_error:
                            return MetricResult(value=None, reason=validation_error)

                        return result

                    except Exception as e:
                        # Handle errors gracefully
                        error_msg = f"Error executing metric {self.name}: {str(e)}"
                        return MetricResult(value=None, reason=error_msg)

                def score(self, llm: t.Optional[BaseRagasLLM] = None, **kwargs):
                    """Synchronous scoring method."""
                    return self._execute_metric(llm, is_async_execution=False, **kwargs)

                async def ascore(self, llm: t.Optional[BaseRagasLLM] = None, **kwargs):
                    """Asynchronous scoring method."""
                    # Prepare function arguments based on what the function expects
                    func_kwargs = kwargs.copy()
                    func_args = []

                    if expects_llm:
                        func_args.append(llm)
                    if expects_prompt:
                        func_args.append(self.prompt)

                    if is_async:
                        # For async functions, await the result
                        result = await func(*func_args, **func_kwargs)
                    else:
                        # For sync functions, run normally
                        result = self._run_sync_in_async(
                            func, *func_args, **func_kwargs
                        )

                    # Ensure result is a MetricResult
                    if not isinstance(result, MetricResult):
                        raise ValueError(
                            f"Custom metric function must return MetricResult, got {type(result)}"
                        )

                    # Validate the result based on metric type
                    validation_error = self._validate_result_value(result.value)
                    if validation_error:
                        return MetricResult(value=None, reason=validation_error)

                    return result

            # Create the metric instance with all parameters
            metric_instance = CustomMetric(name=metric_name, **metric_params)

            # Preserve metadata
            metric_instance.__name__ = metric_name
            metric_instance.__doc__ = func.__doc__

            return metric_instance

        return decorator

    return decorator_factory
