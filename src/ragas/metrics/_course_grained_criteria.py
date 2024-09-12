from __future__ import annotations

import logging
import typing as t
from collections import Counter
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

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


logger = logging.getLogger(__name__)


class CourseGrainedOutput(BaseModel):
    reason: str = Field(description="Reason for the scoring")
    score: int = Field(description="The score for the submission")


class CourseGrainedInput(BaseModel):
    user_input: str = Field(description="The input to the model")
    response: str = Field(description="The response from the model")
    criteria: str = Field(description="The criteria to evaluate the response")


class MultiTurnCourseGrainedInput(BaseModel):
    user_input: str = Field(description="The input to the model")
    criteria: str = Field(description="The criteria to evaluate the response")


class SingleTurnCourseGrainedPrompt(
    PydanticPrompt[CourseGrainedInput, CourseGrainedOutput]
):
    instruction = "Given a input and response. Evaluate and score the submission only using the given criteria."
    input_model = CourseGrainedInput
    output_model = CourseGrainedOutput
    examples = [
        (
            CourseGrainedInput(
                user_input="Who was the director of Los Alamos Laboratory?",
                response="Einstein was the director of Los Alamos Laboratory.",
                criteria="Score responses in range of 0 to 5 based on factors such as grammar, relevance, and coherence.",
            ),
            CourseGrainedOutput(
                reason="The response is grammatically correct and relevant to the input.",
                score=5,
            ),
        )
    ]


class MultiTurnCourseGrainedPrompt(
    PydanticPrompt[MultiTurnCourseGrainedInput, CourseGrainedOutput]
):
    instruction = "Given an interaction between Human, AI and Tools evaluate and score the interaction using the given criteria."
    input_model = MultiTurnCourseGrainedInput
    output_model = CourseGrainedOutput
    examples = [
        (
            MultiTurnCourseGrainedInput(
                user_input="""Human: Hey, book a table at the nearest best Chinese restaurant for 8:00pm\nAI: Sure, let me find the best options for you.\nTools:\n  restaurant_search: {'cuisine': 'Chinese', 'time': '8:00pm'}\nToolOutput: Found a few options: 1. Golden Dragon, 2. Jade Palace\nAI: I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?\nHuman: Let's go with Golden Dragon.\nAI: Great choice! I'll book a table for 8:00pm at Golden Dragon.\nTools:\n  restaurant_book: {'name': 'Golden Dragon', 'time': '8:00pm'}\nToolOutput: Table booked at Golden Dragon for 8:00pm.\nAI: Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!\nHuman: thanks""",
                criteria="Score the interaction in range of 0 to 5 based on factors such as helpfulness, coherence, and relevance.",
            ),
            CourseGrainedOutput(
                reason="The interaction is coherent and relevant to the user's request.",
                score=5,
            ),
        )
    ]


class CourseGrainedOutout(BaseModel):
    reason: str = Field(description="Reason for the score")
    score: int = Field(description="The score for the submission")


class CourseGrainedWithoutReferenceInput(BaseModel):
    user_input: str = Field(description="The input to the model")
    response: str = Field(description="The response from the model")
    criteria: str = Field(description="The criteria to evaluate the response")


class CourseGrainedWithReferenceInput(BaseModel):
    user_input: str = Field(description="The input to the model")
    response: str = Field(description="The response from the model")
    reference: str = Field(description="The reference response")
    criteria: str = Field(description="The criteria to evaluate the response")


@dataclass
class CourseGrainedScore(MetricWithLLM, SingleTurnMetric, MultiTurnMetric):
    """
    Judges the submission to give binary results using the criteria specified
    in the metric definition.

    Attributes
    ----------
    name: str
        name of the metrics
    definition: str
        criteria to judge the submission, example "Is the submission spreading
        fake information?"
    strictness: int
        The number of times self consistency checks is made. Final judgement is
        made using majority vote.
    """

    name: str = field(default="", repr=True)  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            }
        }
    )
    single_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: SingleTurnCourseGrainedPrompt()
    )
    multi_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: MultiTurnCourseGrainedPrompt()
    )
    definition: str = field(default="", repr=True)
    strictness: int = field(default=1, repr=False)
    max_retries: int = 1

    def __post_init__(self: t.Self):
        if self.name == "":
            raise ValueError("Expects a name")
        if self.definition == "":
            raise ValueError("Expects definition")

        # ensure odd number of checks to avoid tie in majority vote.
        self.strictness = (
            self.strictness if self.strictness % 2 != 0 else self.strictness + 1
        )

    def _compute_score(
        self, safe_loaded_responses: t.List[CourseGrainedOutput]
    ) -> float:
        if self.strictness > 1:
            score = Counter(
                [item.verdict for item in safe_loaded_responses]
            ).most_common(1)[0][0]
        else:
            score = safe_loaded_responses[0].score

        return score

    async def _single_turn_ascore(
        self: t.Self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "set LLM before use"

        user_input, context, response = (
            row["user_input"],
            row.get("retrieved_contexts"),
            row["response"],
        )

        if context is not None:
            if isinstance(context, list):
                context = "\n".join(context)
            user_input = f"Question: {user_input} Answer using context: {context}"

        prompt_input = CourseGrainedInput(
            user_input=user_input,
            response=response,
            criteria=self.definition,
        )

        response = await self.single_turn_prompt.generate(
            prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )

        return self._compute_score([response])

    async def _multi_turn_ascore(
        self: t.Self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.rubrics is not None, "Rubrics are not set"
        assert sample.reference is not None, "Reference is not set"

        interaction = sample.pretty_repr()
        reference = sample.reference
        prompt_input = CourseGrainedInput(
            user_input=interaction,
            response=reference,
            criteria=self.definition,
        )
        response = await self.multi_turn_prompt.generate(
            prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )
        return self._compute_score([response])
