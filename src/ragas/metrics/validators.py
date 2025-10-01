"""Validation mixins for different metric types."""

__all__ = [
    "DiscreteValidator",
    "NumericValidator",
    "RankingValidator",
    "AllowedValuesType",
    "get_validator_for_allowed_values",
    "get_metric_type_name",
]

import typing as t
from abc import ABC

# Type alias for all possible allowed_values types across different metric types
AllowedValuesType = t.Union[t.List[str], t.Tuple[float, float], range, int]


class BaseValidator(ABC):
    """Base validator mixin with common validation interface."""

    name: str
    # Note: allowed_values is now inherited from SimpleBaseMetric base class

    def validate_result_value(self, result_value: t.Any) -> t.Optional[str]:
        """
        Validate result value based on metric type constraints.

        Args:
            result_value: The value to validate

        Returns:
            Error message if validation fails, None if validation passes
        """
        raise NotImplementedError


class DiscreteValidator(BaseValidator):
    """Mixin for discrete metric validation with allowed string values."""

    allowed_values: t.List[str]

    def validate_result_value(self, result_value: t.Any) -> t.Optional[str]:
        """Validate that result value is in the allowed discrete values."""
        if not isinstance(self.allowed_values, list):
            return None  # Not a discrete metric

        if result_value not in self.allowed_values:
            return f"Metric {self.name} returned '{result_value}' but expected one of {self.allowed_values}"
        return None


class NumericValidator(BaseValidator):
    """Mixin for numeric metric validation with value ranges."""

    allowed_values: t.Union[t.Tuple[float, float], range]

    def validate_result_value(self, result_value: t.Any) -> t.Optional[str]:
        """Validate that result value is within the numeric range."""
        if not isinstance(self.allowed_values, (tuple, range)):
            return None  # Not a numeric metric

        if not isinstance(result_value, (int, float)):
            return f"Metric {self.name} returned '{result_value}' but expected a numeric value"

        if isinstance(self.allowed_values, tuple):
            min_val, max_val = self.allowed_values
            if not (min_val <= result_value <= max_val):
                return f"Metric {self.name} returned {result_value} but expected value in range {self.allowed_values}"
        elif isinstance(self.allowed_values, range):
            if result_value not in self.allowed_values:
                return f"Metric {self.name} returned {result_value} but expected value in range {self.allowed_values}"
        return None


class RankingValidator(BaseValidator):
    """Mixin for ranking metric validation with expected list lengths."""

    allowed_values: int

    def validate_result_value(self, result_value: t.Any) -> t.Optional[str]:
        """Validate that result value is a list with expected length."""
        if not isinstance(self.allowed_values, int):
            return None  # Not a ranking metric

        if not isinstance(result_value, list):
            return f"Metric {self.name} returned '{result_value}' but expected a list"
        if len(result_value) != self.allowed_values:
            return f"Metric {self.name} returned list of length {len(result_value)} but expected {self.allowed_values} items"
        return None


def get_validator_for_allowed_values(
    allowed_values: AllowedValuesType,
) -> t.Type[BaseValidator]:
    """
    Get the appropriate validator class based on allowed_values type.

    Args:
        allowed_values: The allowed_values to determine validator type

    Returns:
        The appropriate validator class
    """
    if isinstance(allowed_values, list):
        return DiscreteValidator
    elif isinstance(allowed_values, (tuple, range)):
        return NumericValidator
    elif isinstance(allowed_values, int):
        return RankingValidator
    else:
        # Default to discrete if unclear
        return DiscreteValidator


def get_metric_type_name(allowed_values: AllowedValuesType) -> str:
    """Get the metric type name based on allowed_values type."""
    if isinstance(allowed_values, list):
        return "DiscreteMetric"
    elif isinstance(allowed_values, (tuple, range)):
        return "NumericMetric"
    elif isinstance(allowed_values, int):
        return "RankingMetric"
    else:
        return "CustomMetric"
