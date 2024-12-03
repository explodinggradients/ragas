from __future__ import annotations

import logging
import typing as t

from pydantic import BaseModel, Field

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    MultiTurnMetric,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms import BaseRagasLLM

logger = logging.getLogger(__name__)


DEFAULT_REFERENCE_FREE_RUBRICS = {
    "score1_description": "The response is incorrect or does not answer the question.",
    "score2_description": "The response is partially correct but may include errors or incomplete information.",
    "score3_description": "The response is generally correct but lacks clarity or completeness.",
    "score4_description": "The response is correct and clear, with minor issues or missing details.",
    "score5_description": "The response is completely accurate, clear, and answers the question directly.",
}

DEFAULT_WITH_REFERENCE_RUBRICS = {
    "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
    "score2_description": "The response partially matches the ground truth but includes significant errors, omissions, or irrelevant information.",
    "score3_description": "The response generally aligns with the ground truth but may lack detail, clarity, or have minor inaccuracies.",
    "score4_description": "The response is mostly accurate and aligns well with the ground truth, with only minor issues or missing details.",
    "score5_description": "The response is fully accurate, aligns completely with the ground truth, and is clear and detailed.",
}


class ScoreFeedback(BaseModel):
    feedback: str = Field(..., description="The feedback for the response")
    score: int = Field(..., description="The score given to the response")


class SingleTurnInputWithoutRubric(BaseModel):
    user_input: t.Optional[str] = Field(
        description="The input to the llm system", default=None
    )
    response: t.Optional[str] = Field(
        description="The response from the llm system", default=None
    )
    retrieved_contexts: t.Optional[t.List[str]] = Field(
        description="The retrieved contexts from the llm system", default=None
    )
    reference_contexts: t.Optional[t.List[str]] = Field(
        description="The reference contexts for the evaluation", default=None
    )
    reference: t.Optional[str] = Field(
        description="The reference answer for evaluation", default=None
    )


class MultiTurnInputWithoutRubric(BaseModel):
    user_input: t.Optional[str] = Field(description="The user input", default=None)
    reference: t.Optional[str] = Field(
        description="The reference answer for evaluation", default=None
    )


class SingleTurnPrompt(PydanticPrompt[SingleTurnInputWithoutRubric, ScoreFeedback]):
    instruction = ""  # this will be set in the constructor
    input_model = SingleTurnInputWithoutRubric
    output_model = ScoreFeedback


class MultiTurnPrompt(PydanticPrompt[MultiTurnInputWithoutRubric, ScoreFeedback]):
    instruction = ""  # this will be set in the constructor
    input_model = MultiTurnInputWithoutRubric
    output_model = ScoreFeedback


class RubricsScore(MetricWithLLM, SingleTurnMetric, MultiTurnMetric):
    def __init__(
        self,
        name: str = "domain_specific_rubrics",
        rubrics: t.Dict[str, str] = DEFAULT_REFERENCE_FREE_RUBRICS,
        llm: t.Optional[BaseRagasLLM] = None,
        required_columns: t.Optional[t.Dict[MetricType, t.Set[str]]] = None,
        output_type: t.Optional[MetricOutputType] = MetricOutputType.DISCRETE,
        single_turn_prompt: t.Optional[PydanticPrompt] = None,
        multi_turn_prompt: t.Optional[PydanticPrompt] = None,
        max_retries: int = 1,
    ):
        self.rubrics = rubrics
        self.single_turn_scoring_prompt = single_turn_prompt or SingleTurnPrompt()
        self.multi_turn_scoring_prompt = multi_turn_prompt or MultiTurnPrompt()
        self.max_retries = max_retries
        self._required_columns = required_columns or {
            MetricType.SINGLE_TURN: {
                "user_input:optional",
                "response:optional",
                "retrieved_contexts:optional",
                "reference:optional",
                "reference_contexts:optional",
            },
            MetricType.MULTI_TURN: {
                "user_input:optional",
                "reference:optional",
            },
        }
        super().__init__(
            name=name,
            llm=llm,
            _required_columns=self._required_columns,
            output_type=output_type,
        )

    def __repr__(self) -> str:
        return f"{self.name}(required_columns={self.required_columns}, llm={self.llm}), rubrics={self.rubrics}"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await self._ascore(sample.to_dict(), callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        user_input = row.get("user_input")
        reference = row.get("reference")
        reference_contexts = row.get("reference_contexts")
        response = row.get("response")
        retrieved_contexts = row.get("retrieved_contexts")

        prompt_input = SingleTurnInputWithoutRubric(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
            reference_contexts=reference_contexts,
        )
        output = await self.single_turn_scoring_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )
        return output.score

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"

        interaction = sample.pretty_repr()
        prompt_input = MultiTurnInputWithoutRubric(
            user_input=interaction,
        )
        output = await self.multi_turn_scoring_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )
        return output.score
