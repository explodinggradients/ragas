import warnings

from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample

# Issue deprecation warning when module is imported
warnings.warn(
    "The 'ragas.testset' module is deprecated and will be removed in a future version. "
    "Testset generation functionality will no longer be supported. "
    "Please consider using alternative tools for synthetic test data generation.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TestsetGenerator",
    "Testset",
    "TestsetSample",
]
