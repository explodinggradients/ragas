from __future__ import annotations

import logging
import typing as t
from collections import Counter
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
    from langchain_core.callbacks.base import Callbacks


logger = logging.getLogger(__name__)


class SimpleCriteriaOutput(BaseModel):
    reason: str = Field(description="Reason for the scoring")
    score: int = Field(description="The score for the submission")


class SingleTurnSimpleCriteriaInput(BaseModel):
    user_input: str = Field(description="The input to the model")
    response: str = Field(description="The response from the model")
    criteria: str = Field(description="The criteria to evaluate the response")


class SingleTurnSimpleCriteriaWithReferenceInput(SingleTurnSimpleCriteriaInput):
    reference: str = Field(description="The reference response")


class MultiTurnSimpleCriteriaInput(BaseModel):
    user_input: str = Field(description="The input to the model")
    criteria: str = Field(description="The criteria to evaluate the response")


class MultiTurnSimpleCriteriaWithReferenceInput(MultiTurnSimpleCriteriaInput):
    reference: str = Field(description="The reference response")


class SingleTurnSimpleCriteriaPrompt(
    PydanticPrompt[SingleTurnSimpleCriteriaInput, SimpleCriteriaOutput]
):
    instruction = "Given a input and response. Evaluate and score the response only using the given criteria."
    input_model = SingleTurnSimpleCriteriaInput
    output_model = SimpleCriteriaOutput
    examples = [
        (
            SingleTurnSimpleCriteriaInput(
                user_input="Who was the director of Los Alamos Laboratory?",
                response="Einstein was the director of Los Alamos Laboratory.",
                criteria="Score responses in range of 0 to 5 based on factors such as grammar, relevance, and coherence.",
            ),
            SimpleCriteriaOutput(
                reason="The response is grammatically correct and relevant to the input.",
                score=5,
            ),
        )
    ]


class SingleTurnSimpleCriteriaWithReferencePrompt(
    PydanticPrompt[SingleTurnSimpleCriteriaWithReferenceInput, SimpleCriteriaOutput]
):
    instruction = "Given a input, system response and reference. Evaluate and score the response against the reference only using the given criteria."
    input_model = SingleTurnSimpleCriteriaWithReferenceInput
    output_model = SimpleCriteriaOutput
    examples = [
        (
            SingleTurnSimpleCriteriaWithReferenceInput(
                user_input="Who was the director of Los Alamos Laboratory?",
                response="Einstein was the director of Los Alamos Laboratory.",
                reference="The director of Los Alamos Laboratory was J. Robert Oppenheimer.",
                criteria="Score responses in range of 0 (low) to 5 (high) based similarity with reference.",
            ),
            SimpleCriteriaOutput(
                reason="The response and reference have two very different answers.",
                score=0,
            ),
        )
    ]


class MultiTurnSimpleCriteriaPrompt(
    PydanticPrompt[MultiTurnSimpleCriteriaInput, SimpleCriteriaOutput]
):
    instruction = "Given an interaction between Human, AI and Tools evaluate and score the interaction using the given criteria."
    input_model = MultiTurnSimpleCriteriaInput
    output_model = SimpleCriteriaOutput
    examples = [
        (
            MultiTurnSimpleCriteriaInput(
                user_input="""Human: Hey, book a table at the nearest best Chinese restaurant for 8:00pm\nAI: Sure, let me find the best options for you.\nTools:\n  restaurant_search: {'cuisine': 'Chinese', 'time': '8:00pm'}\nToolOutput: Found a few options: 1. Golden Dragon, 2. Jade Palace\nAI: I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?\nHuman: Let's go with Golden Dragon.\nAI: Great choice! I'll book a table for 8:00pm at Golden Dragon.\nTools:\n  restaurant_book: {'name': 'Golden Dragon', 'time': '8:00pm'}\nToolOutput: Table booked at Golden Dragon for 8:00pm.\nAI: Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!\nHuman: thanks""",
                criteria="Score the interaction in range of 0 to 5 based on factors such as helpfulness, coherence, and relevance.",
            ),
            SimpleCriteriaOutput(
                reason="The interaction is coherent and relevant to the user's request.",
                score=5,
            ),
        )
    ]


class MultiTurnSimpleCriteriaWithReferencePrompt(
    PydanticPrompt[MultiTurnSimpleCriteriaWithReferenceInput, SimpleCriteriaOutput]
):
    instruction = "Given an interaction between Human, AI and Tools evaluate and score the interaction using the given criteria."
    input_model = MultiTurnSimpleCriteriaWithReferenceInput
    output_model = SimpleCriteriaOutput
    examples = [
        (
            MultiTurnSimpleCriteriaWithReferenceInput(
                user_input="""Human: Hey, book a table at the nearest best Chinese restaurant for 8:00pm\nAI: Sure, let me find the best options for you.\nTools:\n  restaurant_search: {'cuisine': 'Chinese', 'time': '8:00pm'}\nToolOutput: Found a few options: 1. Golden Dragon, 2. Jade Palace\nAI: I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?\nHuman: Let's go with Golden Dragon.\nAI: Great choice! I'll book a table for 8:00pm at Golden Dragon.\nTools:\n  restaurant_book: {'name': 'Golden Dragon', 'time': '8:00pm'}\nToolOutput: Table booked at Golden Dragon for 8:00pm.\nAI: Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!\nHuman: thanks""",
                reference="The AI successfully books a table at the nearest best Chinese restaurant for 8:00pm, providing the user with options and confirming the booking.",
                criteria="Score the interaction in range of 0 to 5 based on factors such as helpfulness, coherence, and relevance.",
            ),
            SimpleCriteriaOutput(
                reason="The interaction is coherent and relevant to the user's request.",
                score=5,
            ),
        )
    ]


class SimpleCriteriaOutout(BaseModel):
    reason: str = Field(description="Reason for the score")
    score: int = Field(description="The score for the submission")


class SimpleCriteriaWithoutReferenceInput(BaseModel):
    user_input: str = Field(description="The input to the model")
    response: str = Field(description="The response from the model")
    criteria: str = Field(description="The criteria to evaluate the response")


@dataclass
class SimpleCriteriaScoreWithoutReference(
    MetricWithLLM, SingleTurnMetric, MultiTurnMetric
):
    """
    Judges the submission to give binary results using the criteria specified
    in the metric definition.

    Attributes
    ----------
    name: str
        name of the metrics
    definition: str
        criteria to score the submission
    strictness: int
        The number of times self consistency checks is made. Final judgement is
        made using majority vote.
    """

    name: str = field(default="", repr=True)
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            },
            MetricType.MULTI_TURN: {
                "user_input",
            },
        }
    )
    single_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: SingleTurnSimpleCriteriaPrompt()
    )
    multi_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: MultiTurnSimpleCriteriaPrompt()
    )
    definition: str = field(default="", repr=True)
    strictness: int = field(default=1, repr=False)
    max_retries: int = 1

    def __post_init__(self):
        if self.name == "":
            raise ValueError("Expects a name")
        if self.definition == "":
            raise ValueError("Expects definition")

        # ensure odd number of checks to avoid tie in majority vote.
        self.strictness = (
            self.strictness if self.strictness % 2 != 0 else self.strictness + 1
        )

    def _compute_score(
        self, safe_loaded_responses: t.List[SimpleCriteriaOutput]
    ) -> float:
        if self.strictness > 1:
            score = Counter([item.score for item in safe_loaded_responses]).most_common(
                1
            )[0][0]
        else:
            score = safe_loaded_responses[0].score

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
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

        prompt_input = SingleTurnSimpleCriteriaInput(
            user_input=user_input,
            response=response,
            criteria=self.definition,
        )

        response = await self.single_turn_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )

        return self._compute_score([response])

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.reference is not None, "Reference is not set"

        interaction = sample.pretty_repr()
        prompt_input = MultiTurnSimpleCriteriaInput(
            user_input=interaction,
            criteria=self.definition,
        )
        response = await self.multi_turn_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )
        return self._compute_score([response])


@dataclass
class SimpleCriteriaScoreWithReference(SimpleCriteriaScoreWithoutReference):
    name: str = field(default="", repr=True)
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "reference",
            },
            MetricType.MULTI_TURN: {
                "user_input",
                "reference",
            },
        }
    )
    single_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: SingleTurnSimpleCriteriaWithReferencePrompt()
    )
    multi_turn_prompt: PydanticPrompt = field(
        default_factory=lambda: MultiTurnSimpleCriteriaWithReferencePrompt()
    )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.user_input is not None, "User input is not set"
        assert sample.reference is not None, "Reference is not set"
        assert sample.response is not None, "Response is not set"

        prompt_input = SingleTurnSimpleCriteriaWithReferenceInput(
            user_input=sample.user_input,
            response=sample.response,
            reference=sample.reference,
            criteria=self.definition,
        )

        response = await self.single_turn_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )

        return self._compute_score([response])

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.user_input is not None, "User input is not set"
        assert sample.reference is not None, "Reference is not set"

        interaction = sample.pretty_repr()
        prompt_input = MultiTurnSimpleCriteriaWithReferenceInput(
            user_input=interaction,
            reference=sample.reference,
            criteria=self.definition,
        )

        response = await self.multi_turn_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )

        return self._compute_score([response])

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        sample = SingleTurnSample(**row)
        return await self._single_turn_ascore(sample, callbacks)
