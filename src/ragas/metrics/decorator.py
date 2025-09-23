"""decorator factory for creating custom metrics"""

__all__ = ["create_metric_decorator"]

import asyncio
import inspect
import typing as t
import warnings
from dataclasses import dataclass
from typing import get_args, get_origin, get_type_hints

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

                def _check_type(self, value, expected_type):
                    """Check if value matches expected type, handling Union, Optional, etc."""
                    if expected_type == inspect.Parameter.empty:
                        return True  # No type hint, assume valid

                    # Handle Optional (Union[X, None])
                    origin = get_origin(expected_type)
                    if origin is t.Union:
                        args = get_args(expected_type)
                        # For Union types (including Optional), check if value matches any of the types
                        return any(isinstance(value, arg) for arg in args)

                    # Handle basic types
                    if isinstance(expected_type, type):
                        return isinstance(value, expected_type)

                    # For complex types, do basic check
                    return True

                def _create_positional_error(self, args: tuple, kwargs: dict) -> str:
                    """Create error message for positional arguments."""
                    func_param_names = [
                        p for p in sig.parameters.keys() if p not in ["llm", "prompt"]
                    ]

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

                def _create_missing_args_error(self, missing: list) -> str:
                    """Create error message for missing required arguments."""
                    msg = f"\n❌ Missing required arguments for {self.name}:\n\n"

                    try:
                        type_hints = get_type_hints(func)
                    except (NameError, AttributeError):
                        type_hints = {}

                    msg += "   Required:\n"
                    for name in missing:
                        param = sig.parameters[name]
                        type_hint = type_hints.get(name, param.annotation)
                        type_str = self._format_type_name(type_hint)
                        msg += f"     - {name}: {type_str}\n"

                    msg += f"\n   Example: {self.name}.score("
                    examples = []
                    for name in missing:
                        param = sig.parameters[name]
                        type_hint = type_hints.get(name, param.annotation)
                        if type_hint is str or (
                            hasattr(type_hint, "__name__")
                            and type_hint.__name__ == "str"
                        ):
                            examples.append(f'{name}="example"')
                        elif type_hint is float or (
                            hasattr(type_hint, "__name__")
                            and type_hint.__name__ == "float"
                        ):
                            examples.append(f"{name}=0.5")
                        elif type_hint is int or (
                            hasattr(type_hint, "__name__")
                            and type_hint.__name__ == "int"
                        ):
                            examples.append(f"{name}=1")
                        elif (
                            hasattr(type_hint, "__name__")
                            and "list" in type_hint.__name__.lower()
                        ):
                            examples.append(f'{name}=["item1", "item2"]')
                        else:
                            examples.append(f"{name}=...")

                    msg += ", ".join(examples) + ")"
                    return msg

                def _create_type_error(self, type_errors: list) -> str:
                    """Create error message for type validation errors."""
                    msg = f"\n❌ Type mismatch in {self.name}:\n\n"
                    for error in type_errors:
                        msg += f"   {error}\n"
                    return msg

                def _format_type_name(self, type_hint):
                    """Format type name properly for both simple and generic types."""
                    if type_hint == inspect.Parameter.empty:
                        return "Any"

                    # For generic types (like Optional[str], List[int], Union[str, int]),
                    # we want to show the full representation, not just the base class name
                    if hasattr(type_hint, "__origin__") or hasattr(
                        type_hint, "__args__"
                    ):
                        # This is a generic type, use string representation
                        type_str = str(type_hint)
                        # Clean up the representation for readability
                        type_str = type_str.replace("typing.", "")
                        return type_str

                    # For simple types (int, str, float, etc.), use __name__ if available
                    return getattr(type_hint, "__name__", str(type_hint))

                def _is_optional_type(self, type_hint):
                    """Check if a type hint represents an optional type (Union[X, None])."""
                    if type_hint == inspect.Parameter.empty:
                        return False

                    origin = get_origin(type_hint)
                    if origin is t.Union:
                        args = get_args(type_hint)
                        # Check if one of the union args is NoneType
                        return type(None) in args

                    return False

                def _validate_inputs(self, args: tuple, kwargs: dict):
                    """Validate all inputs and provide helpful error messages."""
                    # 1. Check for positional arguments
                    if args:
                        raise TypeError(self._create_positional_error(args, kwargs))

                    # 2. Warn about unknown arguments first (but continue processing)
                    valid_params = set(sig.parameters.keys())
                    unknown = set(kwargs.keys()) - valid_params

                    if unknown:
                        warnings.warn(
                            f"⚠️  {self.name} received unknown arguments: {', '.join(sorted(unknown))}\n"
                            f"   Valid arguments: {', '.join(sorted(valid_params))}",
                            UserWarning,
                        )

                    # 3. Check for required arguments (only check valid parameters)
                    # A parameter is required if:
                    # - It has no default value AND
                    # - It's not 'llm' or 'prompt' AND
                    # - It's not an Optional type (Union[X, None])
                    try:
                        type_hints = get_type_hints(func)
                    except (NameError, AttributeError):
                        type_hints = {}

                    required_params = []
                    for name, p in sig.parameters.items():
                        if name in ["llm", "prompt"]:
                            continue

                        has_default = p.default != inspect.Parameter.empty
                        is_optional_type = self._is_optional_type(
                            type_hints.get(name, p.annotation)
                        )

                        if not has_default and not is_optional_type:
                            required_params.append(name)

                    # Only check for missing required arguments among valid parameters
                    provided_valid_params = set(kwargs.keys()) - unknown
                    missing = [
                        name
                        for name in required_params
                        if name not in provided_valid_params
                    ]
                    if missing:
                        raise TypeError(self._create_missing_args_error(missing))

                    # 4. Type validation (only for valid parameters)
                    try:
                        type_hints = get_type_hints(func)
                        type_errors = []

                        for name, value in kwargs.items():
                            if name in unknown:
                                continue  # Skip unknown parameters
                            if name in type_hints and name in sig.parameters:
                                expected_type = type_hints[name]
                                if not self._check_type(value, expected_type):
                                    actual_type = type(value).__name__
                                    expected_str = self._format_type_name(expected_type)
                                    type_errors.append(
                                        f"- {name}: expected {expected_str}, got {actual_type} ({repr(value)})"
                                    )

                        if type_errors:
                            raise TypeError(self._create_type_error(type_errors))

                    except (NameError, AttributeError):
                        # Skip type validation if we can't get type hints
                        pass

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

                        # Handle Optional parameters that weren't provided
                        try:
                            type_hints = get_type_hints(func)
                        except (NameError, AttributeError):
                            type_hints = {}

                        for name, param in sig.parameters.items():
                            if name in ["llm", "prompt"]:
                                continue

                            # If parameter is not provided but is Optional, provide None
                            if (
                                name not in func_kwargs
                                and param.default == inspect.Parameter.empty
                                and self._is_optional_type(
                                    type_hints.get(name, param.annotation)
                                )
                            ):
                                func_kwargs[name] = None

                        # Remove unknown arguments from func_kwargs before passing to function
                        valid_params = set(sig.parameters.keys())
                        func_kwargs = {
                            k: v for k, v in func_kwargs.items() if k in valid_params
                        }

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

                def score(self, *args, llm: t.Optional[BaseRagasLLM] = None, **kwargs):
                    """Synchronous scoring method."""
                    # Validate inputs before execution
                    self._validate_inputs(args, kwargs)
                    return self._execute_metric(llm, is_async_execution=False, **kwargs)

                async def ascore(
                    self, *args, llm: t.Optional[BaseRagasLLM] = None, **kwargs
                ):
                    """Asynchronous scoring method."""
                    # Validate inputs before execution
                    self._validate_inputs(args, kwargs)
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
                        # Wrap plain values in MetricResult
                        result = MetricResult(value=result, reason=None)

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
