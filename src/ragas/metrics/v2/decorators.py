"""Decorator factory for creating metrics from functions.

This module provides a decorator that converts functions into metric class instances.
"""

import asyncio
import inspect
import typing as t

from pydantic import Field, PrivateAttr

from ragas.metrics.result import MetricResult
from ragas.metrics.v2.base import V2BaseMetric


def v2_metric(
    *,
    name: t.Optional[str] = None,
    allowed_values: t.Tuple[float, float] = (float("-inf"), float("inf")),
    **metric_params: t.Any,
) -> t.Callable[[t.Callable], t.Any]:
    """
    Decorator for creating metrics from functions.

    Args:
        name: Optional name for the metric (defaults to function name)
        allowed_values: Score range as (min, max) tuple (default: no limits)
        **metric_params: Additional parameters to include as fields

    Returns:
        A decorator that transforms a function into a metric instance

    Note:
        Custom parameters passed via **metric_params are added dynamically at runtime.
        Type checkers may not recognize these attributes. If needed, use type: ignore
        or access them via getattr().

    Example:
        >>> from ragas.metrics.v2 import v2_metric
        >>> from ragas.metrics.result import MetricResult
        >>>
        >>> @v2_metric(name="length_score", allowed_values=(0.0, 100.0))
        >>> async def length_metric(response: str) -> MetricResult:
        ...     score = len(response)
        ...     return MetricResult(value=score, reason=f"Response length: {score}")
        >>>
        >>> result = await length_metric.ascore(response="Hello world")
        >>> print(result.value)  # 11.0
    """

    def decorator(func: t.Callable) -> V2BaseMetric:
        """Transform the function into a metric instance."""
        metric_name = name or func.__name__
        is_async = inspect.iscoroutinefunction(func)
        _allowed_values = allowed_values

        class DecoratorGeneratedMetric(V2BaseMetric):
            """Metric generated from decorated function."""

            name: str = Field(default=metric_name)
            allowed_values: t.Tuple[float, float] = Field(default=_allowed_values)

            _func: t.Callable = PrivateAttr()
            _is_async: bool = PrivateAttr()

            def model_post_init(self, __context):
                """Initialize private attributes after model creation."""
                super().model_post_init(__context)
                self._func = func
                self._is_async = is_async

            async def _ascore_impl(self, **kwargs) -> MetricResult:
                """Execute the wrapped function."""
                if self._is_async:
                    result = await self._func(**kwargs)
                else:
                    result = await asyncio.to_thread(self._func, **kwargs)

                if not isinstance(result, MetricResult):
                    result = MetricResult(value=result, reason=None)

                return result

            def __call__(self, *args, **kwargs):
                """Make the metric callable like the original function."""
                if self._is_async:
                    return self.ascore(**kwargs)
                else:
                    return self.score(**kwargs)

        # Add custom parameters as class-level defaults
        for param_name, param_value in metric_params.items():
            if not hasattr(DecoratorGeneratedMetric, "__annotations__"):
                DecoratorGeneratedMetric.__annotations__ = {}
            DecoratorGeneratedMetric.__annotations__[param_name] = t.Any
            setattr(DecoratorGeneratedMetric, param_name, param_value)

        # Recreate the model to register the new fields properly
        DecoratorGeneratedMetric.model_rebuild()

        metric_instance = DecoratorGeneratedMetric()
        metric_instance.__doc__ = func.__doc__
        metric_instance.__module__ = func.__module__

        return metric_instance

    return decorator


create_v2_metric = v2_metric


def v2_numeric_metric(
    *,
    name: t.Optional[str] = None,
    allowed_values: t.Tuple[float, float] = (0.0, 1.0),
    **metric_params: t.Any,
):
    """
    Decorator for creating numeric metrics.

    Example:
        >>> @v2_numeric_metric(name="similarity", allowed_values=(0.0, 1.0))
        >>> async def similarity_metric(text1: str, text2: str) -> float:
        ...     return 0.85
    """
    return v2_metric(name=name, allowed_values=allowed_values, **metric_params)
