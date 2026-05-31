"""JuryEval Score metric — wraps juryeval LLM-as-Judge as a RAGAS metric."""

import asyncio
import typing as t

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult


class JuryEvalScore(BaseMetric):
    """Score a response using a juryeval LLM-as-Judge.

    Requires ``pip install juryeval[judge]``.

    Usage:
        >>> from ragas.metrics.collections import JuryEvalScore
        >>> metric = JuryEvalScore(judge_type="pointwise", judge_model="gpt-4")
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital.",
        ... )

    Args:
        judge_type: ``"pointwise"`` (default) or ``"pairwise"``.
        judge_model: Model name passed to the juryeval judge (default: ``"gpt-4"``).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        temperature: Sampling temperature (default: ``0.0``).
        max_retries: Retry attempts on API failure (default: ``3``).
    """

    def __init__(
        self,
        judge_type: str = "pointwise",
        judge_model: str = "gpt-4",
        api_key: t.Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        name: str = "juryeval_score",
        **base_kwargs,
    ):
        super().__init__(name=name, **base_kwargs)
        self.judge_type = judge_type
        self.judge_model = judge_model
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries

    async def ascore(
        self,
        user_input: str = "",
        response: str = "",
        reference: t.Optional[str] = None,
    ) -> MetricResult:
        loop = asyncio.get_running_loop()

        def _score():
            if self.judge_type == "pointwise":
                from juryeval import PointwiseJudge
                judge = PointwiseJudge(
                    model=self.judge_model,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                )
                result = judge.score(output=response, question=user_input, reference=reference)
                return result["score"], result.get("reason", "")
            elif self.judge_type == "pairwise":
                from juryeval import PairwiseJudge
                judge = PairwiseJudge(
                    model=self.judge_model,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                )
                result = judge.compare(
                    answer_a=response, answer_b=reference or "", question=user_input
                )
                return result["score"], result.get("reason", "")
            else:
                raise ValueError(f"Unknown judge_type: {self.judge_type}")

        score, reason = await loop.run_in_executor(None, _score)
        return MetricResult(value=score, reason=reason)
