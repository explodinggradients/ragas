from __future__ import annotations

import logging
import typing as t

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


class SingleTurnInput(BaseModel):
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
    rubrics: t.Dict[str, str] = Field(..., description="The rubric")


class MultiTurnInput(BaseModel):
    user_input: t.Optional[str] = Field(description="The user input", default=None)
    reference: t.Optional[str] = Field(
        description="The reference answer for evaluation", default=None
    )
    rubrics: t.Dict[str, str] = Field(..., description="The rubric")


class SingleTurnPrompt(PydanticPrompt[SingleTurnInput, ScoreFeedback]):
    instruction = """Given an user_input (which might contain an input along with it), a response to evaluate, and a score rubric representing evaluation criteria are given.
    1. Write detailed feedback that assesses the quality of the response strictly based on the given score rubric, without evaluating in general.
    2. After writing the feedback, assign a score between 1 and 5, referring to the score rubric."""
    input_model = SingleTurnInput
    output_model = ScoreFeedback
    examples = [
        (
            SingleTurnInput(
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


class MultiTurnPrompt(PydanticPrompt[MultiTurnInput, ScoreFeedback]):
    instruction = """Given an interaction between AI,Human and external Tool as input and reference that's desired outcome that get's a score of 5,and a score rubric representing evaluation criteria are given.
    1. Write detailed feedback that assesses the quality of the responselet  strictly based on the given score rubric, without evaluating in general.
    2. After writing the feedback, assign a score between 1 and 5, referring to the score rubric."""
    input_model = MultiTurnInput
    output_model = ScoreFeedback
    examples = [
        (
            MultiTurnInput(
                user_input="""Human: Hey, book a table at the nearest best Chinese restaurant for 8:00pm\nAI: Sure, let me find the best options for you.\nTools:\n  restaurant_search: {'cuisine': 'Chinese', 'time': '8:00pm'}\nToolOutput: Found a few options: 1. Golden Dragon, 2. Jade Palace\nAI: I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?\nHuman: Let's go with Golden Dragon.\nAI: Great choice! I'll book a table for 8:00pm at Golden Dragon.\nTools:\n  restaurant_book: {'name': 'Golden Dragon', 'time': '8:00pm'}\nToolOutput: Table booked at Golden Dragon for 8:00pm.\nAI: Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!\nHuman: thanks""",
                rubrics=DEFAULT_REFERENCE_FREE_RUBRICS,
            ),
            ScoreFeedback(feedback="", score=5),
        ),
        (
            MultiTurnInput(
                user_input="""Human: Hey, book a table at the nearest best Chinese restaurant for 8:00pm\nAI: Sure, let me find the best options for you.\nTools:\n  restaurant_search: {'cuisine': 'Chinese', 'time': '8:00pm'}\nToolOutput: Found a few options: 1. Golden Dragon, 2. Jade Palace\nAI: I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?\nHuman: Let's go with Golden Dragon.\nAI: Great choice! I'll book a table for 8:00pm at Golden Dragon.\nTools:\n  restaurant_book: {'name': 'Golden Dragon', 'time': '8:00pm'}\nToolOutput: Table booked at Golden Dragon for 8:00pm.\nAI: Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!\nHuman: thanks""",
                reference="The AI successfully books a table at the nearest best Chinese restaurant for 8:00pm, providing the user with options and confirming the booking.",
                rubrics=DEFAULT_WITH_REFERENCE_RUBRICS,
            ),
            ScoreFeedback(
                feedback="The AI successfully books a table at the nearest best Chinese restaurant for 8:00pm, providing the user with options and confirming the booking. The response is clear, accurate, and meets all the criteria for a score of 5 based on the rubric.",
                score=5,
            ),
        ),
    ]


class RubricsScore(MetricWithLLM, SingleTurnMetric, MultiTurnMetric):
    def __init__(
        self,
        name: str = "domain_specific_rubrics",
        rubrics: t.Dict[str, str] = DEFAULT_REFERENCE_FREE_RUBRICS,
        llm: t.Optional[BaseRagasLLM] = None,
        required_columns: t.Optional[t.Dict[MetricType, t.Set[str]]] = None,
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
        super().__init__(name=name, llm=llm, _required_columns=self._required_columns)

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

        prompt_input = SingleTurnInput(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
            reference_contexts=reference_contexts,
            rubrics=self.rubrics,
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
        prompt_input = MultiTurnInput(
            user_input=interaction,
            rubrics=self.rubrics,
        )
        output = await self.multi_turn_scoring_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )
        return output.score
