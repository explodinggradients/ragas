from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    MultiTurnMetric,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

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
    name: str = "rubrics_score_without_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts:optional",
            },
            MetricType.MULTI_TURN: {
                "user_input",
            },
        },
        repr=False,
    )
    rubrics: t.Dict[str, str] = field(
        default_factory=lambda: DEFAULT_REFERENCE_FREE_RUBRICS
    )
    max_retries: int = 1
    single_turn_scoring_prompt: PydanticPrompt[
        SingleTurnWithoutReferenceInput, ScoreFeedback
    ] = field(default_factory=SingleTurnWithoutReferencePrompt, repr=False)
    multi_turn_scoring_prompt: PydanticPrompt[
        MultiTurnWithoutReferenceInput, ScoreFeedback
    ] = field(default_factory=MultiTurnWithoutReferencePrompt, repr=False)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await self._ascore(sample.to_dict(), callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_input = self._create_single_turn_prompt(row)
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
        prompt_input = MultiTurnWithoutReferenceInput(
            user_input=interaction,
            rubrics=self.rubrics,
        )
        output = await self.multi_turn_scoring_prompt.generate(
            data=prompt_input,
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


class SingleTurnWithReferenceInput(BaseModel):
    user_input: str = Field(..., description="The user input")
    response: str = Field(..., description="The response")
    reference: str = Field(..., description="The reference")
    rubrics: t.Dict[str, str] = Field(..., description="The rubric")


class MultiTurnWithReferenceInput(BaseModel):
    user_input: str = Field(..., description="The user input")
    reference: str = Field(..., description="The reference")
    rubrics: t.Dict[str, str] = Field(..., description="The rubric")


class SingleTurnWithReferencePrompt(
    PydanticPrompt[SingleTurnWithReferenceInput, ScoreFeedback]
):
    instruction = """Given user input, response and reference that's desired outcome that get's a score of 5,and a score rubric representing evaluation criteria are given.
    1. Write detailed feedback that assesses the quality of the response strictly based on the given score rubric, without evaluating in general.
    2. After writing the feedback, assign a score between 1 and 5, referring to the score rubric."""
    input_model = SingleTurnWithReferenceInput
    output_model = ScoreFeedback
    examples = [
        (
            SingleTurnWithReferenceInput(
                user_input="What is the capital of France?",
                response="The capital of France is Paris.",
                reference="The capital of France is Paris.",
                rubrics=DEFAULT_WITH_REFERENCE_RUBRICS,
            ),
            ScoreFeedback(
                feedback="The response is accurate and provides the correct answer to the question. The language is clear and concise, making it easy to understand. However, additional details could be included to enhance the response.",
                score=5,
            ),
        )
    ]


class MultiTurnWithReferencePrompt(
    PydanticPrompt[MultiTurnWithReferenceInput, ScoreFeedback]
):
    instruction = """Given an interaction between AI,Human and external Tool as input and reference that's desired outcome that get's a score of 5,and a score rubric representing evaluation criteria are given.
    1. Write detailed feedback that assesses the quality of the responselet  strictly based on the given score rubric, without evaluating in general.
    2. After writing the feedback, assign a score between 1 and 5, referring to the score rubric."""
    input_model = MultiTurnWithReferenceInput
    output_model = ScoreFeedback
    examples = [
        (
            MultiTurnWithReferenceInput(
                user_input="""Human: Hey, book a table at the nearest best Chinese restaurant for 8:00pm\nAI: Sure, let me find the best options for you.\nTools:\n  restaurant_search: {'cuisine': 'Chinese', 'time': '8:00pm'}\nToolOutput: Found a few options: 1. Golden Dragon, 2. Jade Palace\nAI: I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?\nHuman: Let's go with Golden Dragon.\nAI: Great choice! I'll book a table for 8:00pm at Golden Dragon.\nTools:\n  restaurant_book: {'name': 'Golden Dragon', 'time': '8:00pm'}\nToolOutput: Table booked at Golden Dragon for 8:00pm.\nAI: Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!\nHuman: thanks""",
                reference="The AI successfully books a table at the nearest best Chinese restaurant for 8:00pm, providing the user with options and confirming the booking.",
                rubrics=DEFAULT_WITH_REFERENCE_RUBRICS,
            ),
            ScoreFeedback(
                feedback="The AI successfully books a table at the nearest best Chinese restaurant for 8:00pm, providing the user with options and confirming the booking. The response is clear, accurate, and meets all the criteria for a score of 5 based on the rubric.",
                score=5,
            ),
        )
    ]


@dataclass
class RubricsScoreWithReference(MetricWithLLM, SingleTurnMetric, MultiTurnMetric):
    name: str = "rubrics_score_with_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts:optional",
                "reference",
            },
            MetricType.MULTI_TURN: {
                "user_input",
                "reference",
            },
        },
        repr=False,
    )
    rubrics: t.Dict[str, str] = field(
        default_factory=lambda: DEFAULT_WITH_REFERENCE_RUBRICS
    )
    max_retries: int = 1
    single_turn_scoring_prompt: PydanticPrompt[
        SingleTurnWithReferenceInput, ScoreFeedback
    ] = field(default_factory=SingleTurnWithReferencePrompt, repr=False)
    multi_turn_scoring_prompt: PydanticPrompt[
        MultiTurnWithReferenceInput, ScoreFeedback
    ] = field(default_factory=MultiTurnWithReferencePrompt, repr=False)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await self._ascore(sample.to_dict(), callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_input = self._create_single_turn_prompt(row)
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
        row = {"interaction": interaction, "reference": sample.reference}
        prompt_input = self._create_multi_turn_prompt(row)
        output = await self.multi_turn_scoring_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )
        return output.score

    def _create_multi_turn_prompt(self, row: t.Dict) -> MultiTurnWithReferenceInput:
        interaction, reference = row["interaction"], row["reference"]
        return MultiTurnWithReferenceInput(
            user_input=interaction,
            reference=reference,
            rubrics=self.rubrics,
        )

    def _create_single_turn_prompt(self, row: t.Dict) -> SingleTurnWithReferenceInput:
        question, contexts, answer, ground_truth = (
            row["user_input"],
            row.get("retrieved_contexts"),
            row["response"],
            row["reference"],
        )
        if contexts:
            contexts = "\n".join(contexts)
            question = f"{question} answer using context: {contexts}"

        return SingleTurnWithReferenceInput(
            user_input=question,
            response=answer,
            reference=ground_truth,
            rubrics=self.rubrics,
        )
