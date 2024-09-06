from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    MultiTurnMetric,
    SingleTurnMetric,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


class ScoreFeedback(BaseModel):
    feedback: str = Field(..., description="The feedback for the response")
    score: int = Field(..., description="The score given to the response")


class ScoreFeedbackAnswers(BaseModel):
    __root__: t.List[ScoreFeedback]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_score_feedback_output_instructions = get_json_format_instructions(ScoreFeedbackAnswers)
_score_feedback_output_parser = RagasoutputParser(pydantic_object=ScoreFeedbackAnswers)

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


SINGLE_TURN_WITH_REFERENCE_SCORING_PROMPT = Prompt(
    name="prometheus_score",
    output_format_instruction=_score_feedback_output_instructions,
    instruction="""Given an question (which might contain an input along with it), a answer to evaluate, a ground_truth answer that gets a score of 5, and a score rubric representing evaluation criteria are given.
1. Write detailed feedback that assesses the quality of the answer strictly based on the given score rubric, without evaluating in general.
2. After writing the feedback, assign a score between 1 and 5, referring to the score rubric.""",
    examples=[
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "ground_truth": "The capital of France is Paris.",
            "rubrics": DEFAULT_WITH_REFERENCE_RUBRICS,
            "analysis": ScoreFeedbackAnswers.parse_obj(
                [
                    {
                        "feedback": """The response is completely accurate and directly answers the question about the capital of France. It matches the reference answer perfectly and does not contain any errors or omissions. Given the rubric, this response deserves the highest score as it meets all the criteria for accuracy and clarity.""",
                        "score": 5,
                    }
                ]
            ).dicts(),
        }
    ],
    input_keys=["question", "answer", "ground_truth", "rubrics"],
    output_key="analysis",
    language="english",
)


MULTI_TURN_WITH_REFERENCE_SCORING_PROMPT = Prompt(
    name="prometheus_score",
    output_format_instruction=_score_feedback_output_instructions,
    instruction="""Given an interaction between AI,Human and external Tool as input and reference that's desired outcome that get's a score of 5,and a score rubric representing evaluation criteria are given.
1. Write detailed feedback that assesses the quality of the responselet  strictly based on the given score rubric, without evaluating in general.
2. After writing the feedback, assign a score between 1 and 5, referring to the score rubric.""",
    examples=[
        {
            "interaction": """
            Human: Hey, book a table at the nearest best Chinese restaurant for 8:00pm\nAI: Sure, let me find the best options for you.\nTools:\n  restaurant_search: {'cuisine': 'Chinese', 'time': '8:00pm'}\nToolOutput: Found a few options: 1. Golden Dragon, 2. Jade Palace\nAI: I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?\nHuman: Let's go with Golden Dragon.\nAI: Great choice! I'll book a table for 8:00pm at Golden Dragon.\nTools:\n  restaurant_book: {'name': 'Golden Dragon', 'time': '8:00pm'}\nToolOutput: Table booked at Golden Dragon for 8:00pm.\nAI: Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!\nHuman: thanks
            """,
            "reference": "The AI successfully books a table at the nearest best Chinese restaurant for 8:00pm, providing the user with options and confirming the booking.",
            "rubrics": DEFAULT_WITH_REFERENCE_RUBRICS,
            "analysis": ScoreFeedbackAnswers.parse_obj(
                [
                    {
                        "feedback": """The AI successfully books a table at the nearest best Chinese restaurant for 8:00pm, providing the user with options and confirming the booking. The response is clear, accurate, and meets all the criteria for a score of 5 based on the rubric.""",
                        "score": 5,
                    }
                ]
            ).dicts(),
        }
    ],
    input_keys=["interaction", "reference", "rubrics"],
    output_key="analysis",
    language="english",
)


WITHOUT_REFERENCE_SCORING_PROMPT = Prompt(
    name="prometheus_score",
    output_format_instruction=_score_feedback_output_instructions,
    instruction="""Given an question (which might contain an input along with it), a answer to evaluate, and a score rubric representing evaluation criteria are given.
1. Write detailed feedback that assesses the quality of the answer strictly based on the given score rubric, without evaluating in general.
2. After writing the feedback, assign a score between 1 and 5, referring to the score rubric.""",
    examples=[
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "rubrics": DEFAULT_REFERENCE_FREE_RUBRICS,
            "analysis": ScoreFeedbackAnswers.parse_obj(
                [
                    {
                        "feedback": """The response is completely accurate and directly answers the question about the capital of France. It matches the reference answer perfectly and does not contain any errors or omissions. Given the rubric, this response deserves the highest score as it meets all the criteria for accuracy and clarity.""",
                        "score": 5,
                    }
                ]
            ).dicts(),
        }
    ],
    input_keys=["question", "answer", "rubrics"],
    output_key="analysis",
    language="english",
)


@dataclass
class LabelledRubricsScore(MetricWithLLM, SingleTurnMetric, MultiTurnMetric):
    name: str = "labelled_rubrics_score"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference"},
            MetricType.MULTI_TURN: {
                "user_input",
                "reference",
            },
        }
    )
    rubrics: t.Dict[str, str] = field(
        default_factory=lambda: DEFAULT_WITH_REFERENCE_RUBRICS
    )
    single_turn_scoring_prompt: Prompt = field(
        default_factory=lambda: SINGLE_TURN_WITH_REFERENCE_SCORING_PROMPT
    )
    multi_turn_scoring_prompt: Prompt = field(
        default_factory=lambda: MULTI_TURN_WITH_REFERENCE_SCORING_PROMPT
    )
    max_retries: int = 1

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await self._ascore(sample.dict(), callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_value = self._create_single_turn_prompt(row)

        response = await self.llm.generate(prompt_value, callbacks=callbacks)

        parsed_response = await _score_feedback_output_parser.aparse(
            response.generations[0][0].text, prompt_value, self.llm, self.max_retries
        )

        if parsed_response is None:
            return np.nan

        score = parsed_response.dicts()[0]["score"]
        return score

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        
        interaction = sample.pretty_repr()
        row = {"interaction": interaction, "reference": sample.reference}
        prompt = self._create_multi_turn_prompt(row)
        response = await self.llm.generate(prompt, callbacks=callbacks)
        parsed_response = await _score_feedback_output_parser.aparse(
            response.generations[0][0].text, prompt, self.llm, self.max_retries
        )
        if parsed_response is None:
            return np.nan
        score = parsed_response.dicts()[0]["score"]
        return score

    
    def _create_multi_turn_prompt(self, row: t.Dict) -> PromptValue:
        interaction, reference = row["interaction"], row["reference"]
        return self.multi_turn_scoring_prompt.format(
            interaction=interaction, reference=reference, rubrics=self.rubrics
        )

    def _create_single_turn_prompt(self, row: t.Dict) -> PromptValue:
        question, contexts, answer, ground_truth = (
            row["user_input"],
            row["retrieved_contexts"],
            row["response"],
            row["reference"],
        )
        contexts = "\n".join(contexts)
        question = f"{question} answer using context: {contexts}"
        return self.single_turn_scoring_prompt.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            rubrics=self.rubrics,
        )

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "LLM must be set to adapt the metric"
        self.single_turn_scoring_prompt.adapt(language, self.llm, cache_dir)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.single_turn_scoring_prompt.save(cache_dir)


@dataclass
class ReferenceFreeRubricsScore(LabelledRubricsScore, SingleTurnMetric):
    name: str = "reference_free_rubrics_score"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
    )
    rubrics: t.Dict[str, str] = field(
        default_factory=lambda: DEFAULT_REFERENCE_FREE_RUBRICS
    )
    scoring_prompt: Prompt = field(
        default_factory=lambda: WITHOUT_REFERENCE_SCORING_PROMPT
    )
    max_retries: int = 1

    def _create_prompt(self, row: t.Dict) -> PromptValue:
        question, contexts, answer = (
            row["user_input"],
            row["retrieved_contexts"],
            row["response"],
        )
        contexts = "\n".join(contexts)
        question = f"{question} answer using context: {contexts}"
        return self.scoring_prompt.format(
            question=question,
            answer=answer,
            rubrics=self.rubrics,
        )


labelled_rubrics_score = LabelledRubricsScore()
reference_free_rubrics_score = ReferenceFreeRubricsScore()
