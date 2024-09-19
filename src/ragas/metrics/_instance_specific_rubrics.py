from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.experimental.llms.prompt import PydanticPrompt
from ragas.metrics._domain_specific_rubrics import (
    MultiTurnWithoutReferenceInput,
    MultiTurnWithoutReferencePrompt,
    MultiTurnWithReferenceInput,
    SingleTurnWithoutReferenceInput,
    SingleTurnWithoutReferencePrompt,
    SingleTurnWithReferenceInput,
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


@dataclass
class InstanceRubricsWithReference(MetricWithLLM, SingleTurnMetric, MultiTurnMetric):
    name: str = "labelled_rubrics_score"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference", "rubrics"},
            MetricType.MULTI_TURN: {"user_input", "reference", "rubrics"},
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

        user_input, contexts, response, reference, rubrics = (
            row["user_input"],
            row.get("retrieved_contexts"),
            row["response"],
            row["reference"],
            row["rubrics"],
        )
        if contexts is not None:
            contexts = "\n".join(contexts)
            user_input = f"{user_input} answer using context: {contexts}"

        prompt_input = SingleTurnWithReferenceInput(
            user_input=user_input,
            response=response,
            reference=reference,
            rubrics=rubrics,
        )

        response = await self.single_turn_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return response.score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.rubrics is not None, "Rubrics are not set"
        assert sample.reference is not None, "Reference is not set"

        interaction = sample.pretty_repr()
        reference = sample.reference
        rubrics = sample.rubrics
        prompt_input = MultiTurnWithReferenceInput(
            user_input=interaction,
            reference=reference,
            rubrics=rubrics,
        )
        output = await self.multi_turn_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )
        return output.score


@dataclass
class InstanceRubricsScoreWithoutReference(
    MetricWithLLM, SingleTurnMetric, MultiTurnMetric
):
    name: str = "reference_free_rubrics_score"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "rubrics"},
            MetricType.MULTI_TURN: {"user_input", "rubrics"},
        }
    )
    single_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: SingleTurnWithoutReferencePrompt()
    )
    multi_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: MultiTurnWithoutReferencePrompt()
    )
    max_retries: int = 1

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        user_input, contexts, response, rubrics = (
            row["user_input"],
            row.get("retrieved_contexts"),
            row["response"],
            row["rubrics"],
        )
        if contexts is not None:
            contexts = "\n".join(contexts)
            user_input = f"{user_input} answer using context: {contexts}"

        prompt_input = SingleTurnWithoutReferenceInput(
            user_input=user_input,
            response=response,
            rubrics=rubrics,
        )

        response = await self.single_turn_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return response.score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.rubrics is not None, "Rubrics are not set"
        interaction = sample.pretty_repr()
        rubrics = sample.rubrics
        prompt_input = MultiTurnWithoutReferenceInput(
            user_input=interaction,
            rubrics=rubrics,
        )
        output = await self.multi_turn_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )
        return output.score
