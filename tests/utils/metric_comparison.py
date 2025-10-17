"""Utilities for comparing metrics across different implementations.

This module provides tools for comparing legacy and modern metric implementations,
including concurrent execution, statistical analysis, and result export capabilities.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ragas.dataset_schema import SingleTurnSample


@dataclass
class MetricDiffResult:
    """Container for metric comparison results.

    Attributes:
        old_scores: List of scores from the baseline/old metric
        new_scores: List of scores from the new metric
        diffs: List of differences (new - old)
        mean_diff: Mean of differences
        max_diff: Maximum difference
        min_diff: Minimum difference
        std_diff: Standard deviation of differences
        old_mean: Mean of old metric scores
        new_mean: Mean of new metric scores
        old_time: Execution time for old metric (seconds)
        new_time: Execution time for new metric (seconds)
    """

    old_scores: List[float]
    new_scores: List[float]
    diffs: List[float]
    mean_diff: float
    max_diff: float
    min_diff: float
    std_diff: float
    old_mean: float
    new_mean: float
    old_time: float
    new_time: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns:
            DataFrame with columns: old_score, new_score, diff, abs_diff
        """
        return pd.DataFrame(
            {
                "old_score": self.old_scores,
                "new_score": self.new_scores,
                "diff": self.diffs,
                "abs_diff": [abs(d) for d in self.diffs],
            }
        )

    def print_summary(self):
        """Print a formatted summary of the comparison results."""
        print("=" * 60)
        print("METRIC COMPARISON SUMMARY")
        print("=" * 60)
        print("\nScore Statistics:")
        print(f"  Old Metric Mean: {self.old_mean:.4f}")
        print(f"  New Metric Mean: {self.new_mean:.4f}")
        print("\nDifference Statistics (new - old):")
        print(f"  Mean Diff:   {self.mean_diff:.4f}")
        print(f"  Max Diff:    {self.max_diff:.4f}")
        print(f"  Min Diff:    {self.min_diff:.4f}")
        print(f"  Std Dev:     {self.std_diff:.4f}")
        print("\nExecution Time:")
        print(f"  Old Metric:  {self.old_time:.2f}s")
        print(f"  New Metric:  {self.new_time:.2f}s")
        print(
            f"  Speedup:     {self.old_time / self.new_time:.2f}x"
            if self.new_time > 0
            else "  N/A"
        )
        print("=" * 60)


async def run_metric_on_dataset(
    metric: Any,
    dataset: List[Dict[str, Any]],
    metric_type: str = "old",
    max_concurrent: int = 10,
) -> Tuple[List[float], float]:
    """
    Run a metric on a dataset with concurrent processing for better performance.

    This function processes all samples concurrently with a semaphore to limit
    the number of simultaneous API calls, preventing rate limiting issues.

    Args:
        metric: The metric instance (either old or new style)
        dataset: List of dictionaries containing the data samples
        metric_type: "old" for legacy metrics, "new" for collections metrics
        max_concurrent: Maximum number of concurrent requests (default: 10)

    Returns:
        Tuple of (scores list, execution time in seconds)

    Example:
        >>> scores, time = await run_metric_on_dataset(
        ...     metric=my_metric,
        ...     dataset=[{"user_input": "q1", "response": "a1"}],
        ...     metric_type="new",
        ...     max_concurrent=5,
        ... )
    """

    async def score_single_sample(sample_dict: Dict[str, Any]) -> float:
        """Score a single sample using the appropriate metric interface."""
        try:
            if metric_type == "old":
                # Old metrics use SingleTurnSample
                sample = SingleTurnSample(**sample_dict)
                score = await metric._single_turn_ascore(sample, callbacks=None)
            else:
                # New metrics use direct kwargs
                result = await metric.ascore(**sample_dict)
                score = result.value

            return float(score)
        except Exception as e:
            print(f"Error processing sample: {e}")
            return np.nan

    start_time = time.time()

    # Use semaphore to limit concurrent requests (prevents API rate limiting)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_with_limit(sample_dict: Dict[str, Any]) -> float:
        """Score with concurrency control."""
        async with semaphore:
            return await score_single_sample(sample_dict)

    # Process all samples concurrently
    scores = await asyncio.gather(*[score_with_limit(s) for s in dataset])

    execution_time = time.time() - start_time
    return list(scores), execution_time


async def compare_metrics(
    old_metric: Any,
    new_metric: Any,
    dataset: List[Dict[str, Any]],
    old_metric_type: str = "old",
    new_metric_type: str = "new",
    max_concurrent: int = 10,
    parallel_metrics: bool = True,
) -> MetricDiffResult:
    """
    Compare two metrics on the same dataset with optional parallel execution.

    This function runs both metrics on the dataset and computes detailed
    comparison statistics. Metrics can be run in parallel (faster) or
    sequentially (more accurate individual timing).

    Args:
        old_metric: The baseline/old metric instance
        new_metric: The new/updated metric instance
        dataset: List of dictionaries containing the data samples
        old_metric_type: Type identifier for old metric ("old" or "new")
        new_metric_type: Type identifier for new metric ("old" or "new")
        max_concurrent: Maximum number of concurrent requests per metric (default: 10)
        parallel_metrics: If True, run both metrics in parallel. If False, run sequentially
                         for more accurate individual timing (default: True)

    Returns:
        MetricDiffResult containing detailed comparison statistics

    Example:
        >>> result = await compare_metrics(
        ...     old_metric=legacy_metric,
        ...     new_metric=modern_metric,
        ...     dataset=test_data,
        ...     max_concurrent=5,
        ...     parallel_metrics=True,
        ... )
        >>> result.print_summary()
    """
    if parallel_metrics:
        print(
            f"Running both metrics in parallel on {len(dataset)} samples (max {max_concurrent} concurrent)..."
        )

        # Run both metrics concurrently using asyncio.gather
        (old_scores, old_time), (new_scores, new_time) = await asyncio.gather(
            run_metric_on_dataset(old_metric, dataset, old_metric_type, max_concurrent),
            run_metric_on_dataset(new_metric, dataset, new_metric_type, max_concurrent),
        )
    else:
        # Sequential execution for more accurate individual timing
        print(
            f"Running old metric on {len(dataset)} samples (max {max_concurrent} concurrent)..."
        )
        old_scores, old_time = await run_metric_on_dataset(
            old_metric, dataset, old_metric_type, max_concurrent
        )

        print(
            f"Running new metric on {len(dataset)} samples (max {max_concurrent} concurrent)..."
        )
        new_scores, new_time = await run_metric_on_dataset(
            new_metric, dataset, new_metric_type, max_concurrent
        )

    # Calculate differences
    diffs = [new - old for old, new in zip(old_scores, new_scores)]

    return MetricDiffResult(
        old_scores=old_scores,
        new_scores=new_scores,
        diffs=diffs,
        mean_diff=float(np.mean(diffs)),
        max_diff=float(np.max(diffs)),
        min_diff=float(np.min(diffs)),
        std_diff=float(np.std(diffs)),
        old_mean=float(np.mean(old_scores)),
        new_mean=float(np.mean(new_scores)),
        old_time=old_time,
        new_time=new_time,
    )


async def run_metric_on_dataset_with_batching(
    metric: Any,
    dataset: List[Dict[str, Any]],
    metric_type: str = "new",
    batch_size: int = 5,
) -> Tuple[List[float], float]:
    """
    Run metric using batch processing if available (for better performance).

    This function attempts to use the metric's abatch_score method if available,
    which can be more efficient than individual scoring. Falls back to concurrent
    processing if batching is not supported.

    Args:
        metric: The metric instance
        dataset: List of dictionaries containing the data samples
        metric_type: "old" or "new" - old metrics don't support batching
        batch_size: Number of samples per batch (default: 5)

    Returns:
        Tuple of (scores list, execution time in seconds)

    Example:
        >>> scores, time = await run_metric_on_dataset_with_batching(
        ...     metric=my_metric,
        ...     dataset=test_data,
        ...     metric_type="new",
        ...     batch_size=10,
        ... )
    """
    # Check if metric supports batching
    has_batch = hasattr(metric, "abatch_score")

    if not has_batch or metric_type == "old":
        # Fall back to concurrent processing
        print("  Batching not available, using concurrent processing...")
        return await run_metric_on_dataset(metric, dataset, metric_type)

    start_time = time.time()
    all_scores = []

    # Process in batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    print(
        f"  Processing {len(dataset)} samples in {num_batches} batches of {batch_size}..."
    )

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        try:
            results = await metric.abatch_score(batch)
            scores = [r.value for r in results]
            all_scores.extend(scores)
        except Exception as e:
            print(
                f"  Warning: Batch {i // batch_size + 1} failed ({e}), falling back to individual processing..."
            )
            # Fall back to individual processing for this batch
            for sample in batch:
                try:
                    result = await metric.ascore(**sample)
                    all_scores.append(result.value)
                except Exception as e2:
                    print(f"  Error processing sample: {e2}")
                    all_scores.append(np.nan)

    execution_time = time.time() - start_time
    return all_scores, execution_time


def export_comparison_results(
    result: MetricDiffResult,
    dataset: List[Dict[str, Any]],
    filename: str = "metric_comparison_results.csv",
):
    """
    Export comparison results to CSV file.

    The CSV includes all scores, differences, and the original dataset fields,
    plus a summary row with aggregate statistics.

    Args:
        result: MetricDiffResult object containing comparison data
        dataset: Original dataset (to include context in export)
        filename: Output CSV filename (default: "metric_comparison_results.csv")

    Example:
        >>> export_comparison_results(
        ...     result=comparison_result,
        ...     dataset=test_data,
        ...     filename="context_recall_results.csv",
        ... )
    """
    df = result.to_dataframe()

    # Add dataset information
    for key in dataset[0].keys():
        df[key] = [sample.get(key, "") for sample in dataset]

    # Add summary statistics as a separate row
    summary = pd.DataFrame(
        [
            {
                **{
                    key: "SUMMARY" if i == 0 else ""
                    for i, key in enumerate(dataset[0].keys())
                },
                "old_score": result.old_mean,
                "new_score": result.new_mean,
                "diff": result.mean_diff,
                "abs_diff": np.mean([abs(d) for d in result.diffs]),
            }
        ]
    )

    df = pd.concat([df, summary], ignore_index=True)
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")
