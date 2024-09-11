from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np

from ragas.experimental.llms.prompt import PydanticPrompt
from ragas.metrics._domain_specific_rubrics import (
    MultiTurnWithoutReferencePrompt,
    SingleTurnWithoutReferencePrompt,
    SingleTurnWithReferencePrompt,
)
from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    MultiTurnMetric,
    SingleTurnMetric,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


@dataclass
class InstanceRubricsWithReference(MetricWithLLM, SingleTurnMetric, MultiTurnMetric):
    name: str = "labelled_rubrics_score"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference", "rubric"},
            MetricType.MULTI_TURN: {"user_input", "reference", "rubric"},
        }
    )
    single_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: SingleTurnWithReferencePrompt()
    )
    multi_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: MultiTurnWithoutReferencePrompt()
    )

    max_retries: int = 1

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_value = self._create_prompt(row)

        response = await self.llm.generate(prompt_value, callbacks=callbacks)

        parsed_response = await _score_feedback_output_parser.aparse(
            response.generations[0][0].text, prompt_value, self.llm, self.max_retries
        )

        if parsed_response is None:
            return np.nan

        score = parsed_response.dicts()[0]["score"]
        return score

    def _create_prompt(self, row: t.Dict) -> PromptValue:
        question, contexts, answer, ground_truth, rubrics = (
            row["user_input"],
            row.get("retrieved_contexts"),
            row["response"],
            row["reference"],
            row["rubric"],
        )
        if contexts is not None:
            contexts = "\n".join(contexts)
            question = f"{question} answer using context: {contexts}"
        return self.scoring_prompt.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            rubrics=rubrics,
        )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)


@dataclass
class InstanceRubricsScoreWithoutReference(InstanceRubricsWithReference):
    name: str = "reference_free_rubrics_score"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "rubric"},
            MetricType.MULTI_TURN: {"user_input", "rubric"},
        }
    )
    single_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: SingleTurnWithoutReferencePrompt()
    )
    multi_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: MultiTurnWithoutReferencePrompt()
    )
    max_retries: int = 1

    def _create_prompt(self, row: t.Dict) -> PromptValue:
        question, contexts, answer, rubrics = (
            row["user_input"],
            row.get("retrieved_contexts"),
            row["response"],
            row["rubric"],
        )
        if contexts is not None:
            contexts = "\n".join(contexts)
            question = f"{question} answer using context: {contexts}"
        return self.scoring_prompt.format(
            question=question,
            answer=answer,
            rubrics=rubrics,
        )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)
