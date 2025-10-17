"""Shared test utilities for Ragas tests.

This module provides reusable utilities for both pytest tests and Jupyter notebooks,
including LLM setup, embeddings configuration, and common test helpers.
"""

from .llm_setup import (
    check_api_key,
    create_legacy_embeddings,
    create_legacy_llm,
    create_modern_embeddings,
    create_modern_llm,
)
from .metric_comparison import (
    MetricDiffResult,
    compare_metrics,
    export_comparison_results,
    run_metric_on_dataset,
    run_metric_on_dataset_with_batching,
)

__all__ = [
    # LLM and embeddings setup
    "check_api_key",
    "create_legacy_llm",
    "create_modern_llm",
    "create_legacy_embeddings",
    "create_modern_embeddings",
    # Metric comparison utilities
    "MetricDiffResult",
    "compare_metrics",
    "export_comparison_results",
    "run_metric_on_dataset",
    "run_metric_on_dataset_with_batching",
]
