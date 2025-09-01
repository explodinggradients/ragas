"""
Batch evaluation utilities for cost-effective metric evaluation using OpenAI Batch API.

This module provides high-level utilities for running Ragas metrics in batch mode,
offering significant cost savings (up to 50%) for large-scale evaluations.
"""

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.metrics.base import MetricWithLLM

if t.TYPE_CHECKING:
    from ragas.llms.batch_api import BatchResponse

logger = logging.getLogger(__name__)


@dataclass
class BatchEvaluationResult:
    """Results from a batch evaluation job."""

    metric_name: str
    job_id: str
    sample_count: int
    responses: t.List[BatchResponse]
    scores: t.Optional[t.List[t.Optional[float]]] = None
    errors: t.List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the batch job."""
        if not self.responses:
            return 0.0
        successful = sum(1 for resp in self.responses if resp.error is None)
        return successful / len(self.responses)

    @property
    def average_score(self) -> t.Optional[float]:
        """Calculate average score if scores are available."""
        if not self.scores:
            return None
        valid_scores = [s for s in self.scores if s is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None


class BatchEvaluator:
    """High-level interface for batch evaluation using OpenAI Batch API."""

    def __init__(
        self,
        metrics: t.List[MetricWithLLM],
        max_batch_size: int = 1000,
        poll_interval: float = 300.0,  # 5 minutes
        timeout: float = 86400.0,  # 24 hours
    ):
        """
        Initialize batch evaluator.

        Args:
            metrics: List of metrics to evaluate
            max_batch_size: Maximum samples per batch job
            poll_interval: Polling interval for batch job status (seconds)
            timeout: Maximum time to wait for batch completion (seconds)
        """
        self.metrics = metrics
        self.max_batch_size = max_batch_size
        self.poll_interval = poll_interval
        self.timeout = timeout

        # Validate that all metrics support batch evaluation
        for metric in metrics:
            if not metric.supports_batch_evaluation():
                raise ValueError(
                    f"Metric '{metric.name}' does not support batch evaluation. "
                    "Ensure it uses an LLM that supports OpenAI Batch API."
                )

    def evaluate(
        self,
        samples: t.List[t.Union[SingleTurnSample, MultiTurnSample]],
        wait_for_completion: bool = True,
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> t.List[BatchEvaluationResult]:
        """
        Run batch evaluation on samples.

        Args:
            samples: Samples to evaluate
            wait_for_completion: Whether to wait for jobs to complete
            metadata: Optional metadata for batch jobs

        Returns:
            List of batch evaluation results
        """
        if len(samples) > self.max_batch_size:
            raise ValueError(
                f"Sample count {len(samples)} exceeds maximum batch size {self.max_batch_size}"
            )

        results = []
        jobs = []

        # Create batch jobs for each metric
        for metric in self.metrics:
            logger.info(f"Creating batch job for metric: {metric.name}")

            job = metric.create_batch_evaluation_job(
                samples=samples, batch_size=self.max_batch_size, metadata=metadata
            )
            jobs.append((metric, job))

        # Wait for completion if requested
        if wait_for_completion:
            for metric, job in jobs:
                logger.info(
                    f"Waiting for batch job completion: {metric.name} (ID: {job.batch_id})"
                )

                status = job.wait_for_completion(
                    poll_interval=self.poll_interval, timeout=self.timeout
                )

                if status.value == "completed":
                    responses = job.get_results()
                    result = BatchEvaluationResult(
                        metric_name=metric.name,
                        job_id=job.batch_id,
                        sample_count=len(samples),
                        responses=responses,
                    )

                    # Process responses to extract scores if possible
                    # This is a simplified implementation - actual score extraction
                    # would depend on the specific metric and response format
                    try:
                        result.scores = self._extract_scores(metric, responses)
                    except Exception as e:
                        result.errors.append(f"Score extraction failed: {str(e)}")

                    results.append(result)
                else:
                    # Job failed or was cancelled
                    result = BatchEvaluationResult(
                        metric_name=metric.name,
                        job_id=job.batch_id,
                        sample_count=len(samples),
                        responses=[],
                        errors=[f"Batch job failed with status: {status.value}"],
                    )
                    results.append(result)
        else:
            # Return results with pending jobs
            for metric, job in jobs:
                result = BatchEvaluationResult(
                    metric_name=metric.name,
                    job_id=job.batch_id,
                    sample_count=len(samples),
                    responses=[],
                )
                results.append(result)

        return results

    async def aevaluate(
        self,
        samples: t.List[t.Union[SingleTurnSample, MultiTurnSample]],
        wait_for_completion: bool = True,
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> t.List[BatchEvaluationResult]:
        """Async version of evaluate."""
        if len(samples) > self.max_batch_size:
            raise ValueError(
                f"Sample count {len(samples)} exceeds maximum batch size {self.max_batch_size}"
            )

        results = []
        jobs = []

        # Create batch jobs for each metric
        for metric in self.metrics:
            logger.info(f"Creating batch job for metric: {metric.name}")

            job = await metric.acreate_batch_evaluation_job(
                samples=samples, batch_size=self.max_batch_size, metadata=metadata
            )
            jobs.append((metric, job))

        # Wait for completion if requested
        if wait_for_completion:
            for metric, job in jobs:
                logger.info(
                    f"Waiting for batch job completion: {metric.name} (ID: {job.batch_id})"
                )

                status = await job.await_completion(
                    poll_interval=self.poll_interval, timeout=self.timeout
                )

                if status.value == "completed":
                    responses = await job.aget_results()
                    result = BatchEvaluationResult(
                        metric_name=metric.name,
                        job_id=job.batch_id,
                        sample_count=len(samples),
                        responses=responses,
                    )

                    try:
                        result.scores = self._extract_scores(metric, responses)
                    except Exception as e:
                        result.errors.append(f"Score extraction failed: {str(e)}")

                    results.append(result)
                else:
                    result = BatchEvaluationResult(
                        metric_name=metric.name,
                        job_id=job.batch_id,
                        sample_count=len(samples),
                        responses=[],
                        errors=[f"Batch job failed with status: {status.value}"],
                    )
                    results.append(result)
        else:
            for metric, job in jobs:
                result = BatchEvaluationResult(
                    metric_name=metric.name,
                    job_id=job.batch_id,
                    sample_count=len(samples),
                    responses=[],
                )
                results.append(result)

        return results

    def _extract_scores(
        self, metric: MetricWithLLM, responses: t.List[BatchResponse]
    ) -> t.List[t.Optional[float]]:
        """
        Extract scores from batch responses.

        This is a placeholder implementation. In practice, this would need
        to parse the specific response format for each metric and extract
        the numerical scores.
        """
        logger.warning(
            f"Score extraction not implemented for metric: {metric.name}. "
            "Batch responses contain raw LLM outputs that need manual processing."
        )
        return [None] * len(responses)


def create_batch_evaluator(
    metrics: t.List[MetricWithLLM], **kwargs: t.Any
) -> BatchEvaluator:
    """Factory function to create a batch evaluator."""
    return BatchEvaluator(metrics=metrics, **kwargs)


def estimate_batch_cost_savings(
    sample_count: int,
    metrics: t.List[MetricWithLLM],
    regular_cost_per_1k_tokens: float = 0.03,
    batch_discount: float = 0.5,
) -> t.Dict[str, float]:
    """
    Estimate cost savings from using batch API.

    Args:
        sample_count: Number of samples to evaluate
        metrics: List of metrics to run
        regular_cost_per_1k_tokens: Regular API cost per 1K tokens
        batch_discount: Batch API discount (0.5 = 50% savings)

    Returns:
        Dictionary with cost estimates
    """
    estimated_tokens_per_sample = 500
    total_tokens = sample_count * len(metrics) * estimated_tokens_per_sample

    regular_cost = (total_tokens / 1000) * regular_cost_per_1k_tokens
    batch_cost = regular_cost * (1 - batch_discount)
    savings = regular_cost - batch_cost

    return {
        "regular_cost": round(regular_cost, 4),
        "batch_cost": round(batch_cost, 4),
        "savings": round(savings, 4),
        "savings_percentage": round(batch_discount * 100, 1),
        "estimated_tokens": total_tokens,
    }
