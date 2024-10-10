from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from typing import Dict

from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, MetricWithLLM, SingleTurnMetric
from ragas.prompt import PydanticPrompt, StringIO

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

logger = logging.getLogger(__name__)


class ExtractedKeyphrases(BaseModel):
    keyphrases: t.List[str]


class QuestionsGenerated(BaseModel):
    questions: t.List[str]


class AnswersGenerated(BaseModel):
    answers: t.List[str]


class ExtractKeyphrasePrompt(PydanticPrompt[StringIO, ExtractedKeyphrases]):
    name: str = "extract_keyphrases"
    instruction: str = "Extract the keyphrases essential for summarizing the text."
    input_model = StringIO
    output_model = ExtractedKeyphrases
    examples: t.List[t.Tuple[StringIO, ExtractedKeyphrases]] = [
        (
            StringIO(
                text="JPMorgan Chase & Co. is an American multinational finance company headquartered in New York City. It is the largest bank in the United States and the world's largest by market capitalization as of 2023. Founded in 1799, it is a major provider of investment banking services, with US$3.9 trillion in total assets, and ranked #1 in the Forbes Global 2000 ranking in 2023."
            ),
            ExtractedKeyphrases(
                keyphrases=[
                    "JPMorgan Chase & Co.",
                    "American multinational finance company",
                    "headquartered in New York City",
                    "largest bank in the United States",
                    "world's largest bank by market capitalization",
                    "founded in 1799",
                    "major provider of investment banking services",
                    "US$3.9 trillion in total assets",
                    "ranked #1 in Forbes Global 2000 ranking",
                ]
            ),
        )
    ]


class GenerateQuestionsPromptInput(BaseModel):
    text: str
    keyphrases: t.List[str]


class GenerateQuestionsPrompt(
    PydanticPrompt[GenerateQuestionsPromptInput, QuestionsGenerated]
):
    name: str = "generate_questions"
    instruction: str = "Based on the given text and keyphrases, generate closed-ended questions that can be answered with '1' if the question can be answered using the text, or '0' if it cannot. The questions should ALWAYS result in a '1' based on the given text."
    input_model = GenerateQuestionsPromptInput
    output_model = QuestionsGenerated
    examples: t.List[t.Tuple[GenerateQuestionsPromptInput, QuestionsGenerated]] = [
        (
            GenerateQuestionsPromptInput(
                text="JPMorgan Chase & Co. is an American multinational finance company headquartered in New York City. It is the largest bank in the United States and the world's largest by market capitalization as of 2023. Founded in 1799, it is a major provider of investment banking services, with US$3.9 trillion in total assets, and ranked #1 in the Forbes Global 2000 ranking in 2023.",
                keyphrases=[
                    "JPMorgan Chase & Co.",
                    "American multinational finance company",
                    "headquartered in New York City",
                    "largest bank in the United States",
                    "world's largest bank by market capitalization",
                    "founded in 1799",
                    "major provider of investment banking services",
                    "US$3.9 trillion in total assets",
                    "ranked #1 in Forbes Global 2000 ranking",
                ],
            ),
            QuestionsGenerated(
                questions=[
                    "Is JPMorgan Chase & Co. an American multinational finance company?",
                    "Is JPMorgan Chase & Co. headquartered in New York City?",
                    "Is JPMorgan Chase & Co. the largest bank in the United States?",
                    "Is JPMorgan Chase & Co. the world's largest bank by market capitalization as of 2023?",
                    "Was JPMorgan Chase & Co. founded in 1799?",
                    "Is JPMorgan Chase & Co. a major provider of investment banking services?",
                    "Does JPMorgan Chase & Co. have US$3.9 trillion in total assets?",
                    "Was JPMorgan Chase & Co. ranked #1 in the Forbes Global 2000 ranking in 2023?",
                ]
            ),
        )
    ]


class SummaryAndQuestions(BaseModel):
    summary: str
    questions: t.List[str]


class GenerateAnswersPrompt(PydanticPrompt[SummaryAndQuestions, AnswersGenerated]):
    name: str = "generate_answers"
    instruction: str = "Based on the list of close-ended '1' or '0' questions, generate a JSON with key 'answers', which is a list of strings that determines whether the provided summary contains sufficient information to answer EACH question. Answers should STRICTLY be either '1' or '0'. Answer '0' if the provided summary does not contain enough information to answer the question and answer '1' if the provided summary can answer the question."
    input_model = SummaryAndQuestions
    output_model = AnswersGenerated
    examples: t.List[t.Tuple[SummaryAndQuestions, AnswersGenerated]] = [
        (
            SummaryAndQuestions(
                summary="JPMorgan Chase & Co., headquartered in New York City, is the largest bank in the US and the world's largest by market capitalization as of 2023. Founded in 1799, it offers extensive investment, private, asset management, and retail banking services, and has $3.9 trillion in assets, making it the fifth-largest bank globally. It operates the world's largest investment bank by revenue and was ranked #1 in the 2023 Forbes Global 2000.",
                questions=[
                    "Is JPMorgan Chase & Co. an American multinational finance company?",
                    "Is JPMorgan Chase & Co. headquartered in New York City?",
                    "Is JPMorgan Chase & Co. the largest bank in the United States?",
                    "Is JPMorgan Chase & Co. the world's largest bank by market capitalization as of 2023?",
                    "Is JPMorgan Chase & Co. considered systemically important by the Financial Stability Board?",
                    "Was JPMorgan Chase & Co. founded in 1799 as the Chase Manhattan Company?",
                    "Is JPMorgan Chase & Co. a major provider of investment banking services?",
                    "Is JPMorgan Chase & Co. the fifth-largest bank in the world by assets?",
                    "Does JPMorgan Chase & Co. operate the largest investment bank by revenue?",
                    "Was JPMorgan Chase & Co. ranked #1 in the Forbes Global 2000 ranking?",
                    "Does JPMorgan Chase & Co. provide investment banking services?",
                ],
            ),
            AnswersGenerated(
                answers=[
                    "0",
                    "1",
                    "1",
                    "1",
                    "0",
                    "0",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                ]
            ),
        )
    ]


@dataclass
class SummarizationScore(MetricWithLLM, SingleTurnMetric):
    name: str = "summary_score"  # type: ignore
    max_retries: int = 1
    length_penalty: bool = True
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "reference_contexts",
                "response",
            }
        }
    )
    coeff: float = 0.5
    question_generation_prompt: PydanticPrompt = field(
        default_factory=GenerateQuestionsPrompt
    )
    answer_generation_prompt: PydanticPrompt = field(
        default_factory=GenerateAnswersPrompt
    )
    extract_keyphrases_prompt: PydanticPrompt = field(
        default_factory=ExtractKeyphrasePrompt
    )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: Dict, callbacks: Callbacks) -> float:
        text: str = "\n".join(row["reference_contexts"])
        summary: str = row["response"]
        keyphrases = await self._extract_keyphrases(text, callbacks)
        questions = await self._get_questions(text, keyphrases, callbacks)
        answers = await self._get_answers(questions, summary, callbacks)

        scores = {}
        qa_score = self._compute_qa_score(answers)
        scores["qa_score"] = qa_score
        if self.length_penalty:
            conciseness_score = self._compute_conciseness_score(text, summary)
            scores["conciseness_score"] = conciseness_score
        return self._compute_score(scores)

    def _compute_score(self, scores) -> float:
        return (
            scores["qa_score"] * (1 - self.coeff)
            + scores.get("conciseness_score", 0) * self.coeff
        )

    def _compute_qa_score(self, answers: t.List[str]) -> float:
        correct = sum([1 for a in answers if a.lower() == "1"])
        return correct / len(answers)

    def _compute_conciseness_score(self, text, summary) -> float:
        return 1 - min(len(summary), len(text)) / (len(text) + 1e-10)

    async def _extract_keyphrases(self, text: str, callbacks: Callbacks) -> t.List[str]:
        assert self.llm is not None, "LLM is not initialized"

        response: ExtractedKeyphrases = await self.extract_keyphrases_prompt.generate(
            data=StringIO(text=text), llm=self.llm, callbacks=callbacks
        )
        if not response:
            logging.error("No keyphrases generated, unable to calculate the score.")
            return []

        return response.keyphrases

    async def _get_questions(
        self, text: str, keyphrases: list[str], callbacks: Callbacks
    ) -> t.List[str]:
        assert self.llm is not None, "LLM is not initialized"
        response: QuestionsGenerated = await self.question_generation_prompt.generate(
            data=GenerateQuestionsPromptInput(text=text, keyphrases=keyphrases),
            llm=self.llm,
            callbacks=callbacks,
        )
        if not response:
            logging.error("No questions generated, unable to calculate the score.")
            return []

        return response.questions

    async def _get_answers(
        self, questions: t.List[str], summary: str, callbacks: Callbacks
    ) -> t.List[str]:
        assert self.llm is not None, "LLM is not initialized"
        response: AnswersGenerated = await self.answer_generation_prompt.generate(
            data=SummaryAndQuestions(questions=questions, summary=summary),
            llm=self.llm,
            callbacks=callbacks,
        )
        return response.answers


summarization_score = SummarizationScore()
