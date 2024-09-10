from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.experimental.llms.prompt import PydanticPrompt
from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    MultiTurnMetric,
    SingleTurnMetric,
)
from ragas.metrics.domain_specific_rubrics.with_reference import ScoreFeedback

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_FREE_RUBRICS = {
    "score1_description": "The response is incorrect or does not answer the question.",
    "score2_description": "The response is partially correct but may include errors or incomplete information.",
    "score3_description": "The response is generally correct but lacks clarity or completeness.",
    "score4_description": "The response is correct and clear, with minor issues or missing details.",
    "score5_description": "The response is completely accurate, clear, and answers the question directly.",
}


class SingleTurnWithoutReferenceInput(BaseModel):
    user_input: str = Field(..., description="The user input")
    response: str = Field(..., description="The response")
    rubrics: t.Dict[str, str] = Field(..., description="The rubric")


class MultiTurnWithoutReferenceInput(BaseModel):
    user_input: str = Field(..., description="The user input")
    rubrics: t.Dict[str, str] = Field(..., description="The rubric")


class SingleTurnWithoutReferencePrompt(
    PydanticPrompt[SingleTurnWithoutReferenceInput, ScoreFeedback]
):
    instruction = """Given an user_input (which might contain an input along with it), a response to evaluate, and a score rubric representing evaluation criteria are given.
    1. Write detailed feedback that assesses the quality of the response strictly based on the given score rubric, without evaluating in general.
    2. After writing the feedback, assign a score between 1 and 5, referring to the score rubric."""
    input_model = SingleTurnWithoutReferenceInput
    output_model = ScoreFeedback
    examples = [
        (
            SingleTurnWithoutReferenceInput(
                user_input="What is the capital of France?",
                response="The capital of France is Paris.",
                rubrics=DEFAULT_REFERENCE_FREE_RUBRICS,
            ),
            ScoreFeedback(
                feedback="The response is completely accurate and directly answers the question about the capital of France.",
                score=5,
            ),
        )
    ]


class MultiTurnWithoutReferencePrompt(
    PydanticPrompt[MultiTurnWithoutReferenceInput, ScoreFeedback]
):
    instruction = """Given an interaction between AI,Human and external Tool as input and reference that's desired outcome that get's a score of 5,and a score rubric representing evaluation criteria are given.
    1. Write detailed feedback that assesses the quality of the responselet  strictly based on the given score rubric, without evaluating in general.
    2. After writing the feedback, assign a score between 1 and 5, referring to the score rubric."""
    input_model = MultiTurnWithoutReferenceInput
    output_model = ScoreFeedback
    examples = [
        (
            MultiTurnWithoutReferenceInput(
                user_input="""Human: Hey, book a table at the nearest best Chinese restaurant for 8:00pm\nAI: Sure, let me find the best options for you.\nTools:\n  restaurant_search: {'cuisine': 'Chinese', 'time': '8:00pm'}\nToolOutput: Found a few options: 1. Golden Dragon, 2. Jade Palace\nAI: I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?\nHuman: Let's go with Golden Dragon.\nAI: Great choice! I'll book a table for 8:00pm at Golden Dragon.\nTools:\n  restaurant_book: {'name': 'Golden Dragon', 'time': '8:00pm'}\nToolOutput: Table booked at Golden Dragon for 8:00pm.\nAI: Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!\nHuman: thanks""",
                rubrics=DEFAULT_REFERENCE_FREE_RUBRICS,
            ),
            ScoreFeedback(feedback="", score=5),
        )
    ]


@dataclass
class RubricsScoreWithoutReference(MetricWithLLM, SingleTurnMetric, MultiTurnMetric):
    name: str = "rubrics_score_without_reference"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response"},
            MetricType.MULTI_TURN: {
                "user_input",
            },
        }
    )
    rubrics: t.Dict[str, str] = field(
        default_factory=lambda: DEFAULT_REFERENCE_FREE_RUBRICS
    )
    max_retries: int = 1

    def __post_init__(self):
        self.single_turn_scoring_prompt = SingleTurnWithoutReferencePrompt()
        self.multi_turn_scoring_prompt = MultiTurnWithoutReferencePrompt()
        self.rubrics = self.rubrics or DEFAULT_REFERENCE_FREE_RUBRICS

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await self._ascore(sample.dict(), callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_input = self._create_single_turn_prompt(row)
        output = await self.single_turn_scoring_prompt.generate(
            prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )
        return output.score

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"

        interaction = sample.pretty_repr()
        prompt_input = MultiTurnWithoutReferenceInput(
            user_input=interaction,
            rubrics=self.rubrics,
        )
        output = await self.multi_turn_scoring_prompt.generate(
            prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )
        return output.score

    def _create_single_turn_prompt(
        self, row: t.Dict
    ) -> SingleTurnWithoutReferenceInput:
        question, contexts, answer = (
            row["user_input"],
            row.get("retrieved_contexts"),
            row["response"],
        )
        if contexts:
            contexts = "\n".join(contexts)
            question = f"{question} answer using context: {contexts}"

        return SingleTurnWithoutReferenceInput(
            user_input=question,
            response=answer,
            rubrics=self.rubrics,
        )


rubrics_score_without_reference = RubricsScoreWithoutReference()
