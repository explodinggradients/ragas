from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from typing import Dict

from langchain.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt, PromptValue
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

logger = logging.getLogger(__name__)


class ExtractKeyphrasesResponse(BaseModel):
    keyphrases: t.List[str]


class GenerateQuestionsResponse(BaseModel):
    questions: t.List[str]


class GenerateAnswersResponse(BaseModel):
    answers: t.List[str]


_output_instructions_question_generation = get_json_format_instructions(
    pydantic_object=GenerateQuestionsResponse
)
_output_instructions_answer_generation = get_json_format_instructions(
    pydantic_object=GenerateAnswersResponse
)
_output_instructions_keyphrase_extraction = get_json_format_instructions(
    pydantic_object=ExtractKeyphrasesResponse
)
_output_parser_question_generation = RagasoutputParser(
    pydantic_object=GenerateQuestionsResponse
)
_output_parser_answer_generation = RagasoutputParser(
    pydantic_object=GenerateAnswersResponse
)
_output_parser_keyphrase_extraction = RagasoutputParser(
    pydantic_object=ExtractKeyphrasesResponse
)


TEXT_EXTRACT_KEYPHRASES = Prompt(
    name="text_extract_keyphrases",
    instruction="Extract the keyphrases essential for summarizing the text.",
    output_format_instruction=_output_instructions_keyphrase_extraction,
    input_keys=["text"],
    output_key="keyphrases",
    output_type="json",
    examples=[
        {
            "text": """JPMorgan Chase & Co. is an American multinational finance company headquartered in New York City. It is the largest bank in the United States and the world's largest by market capitalization as of 2023. Founded in 1799, it is a major provider of investment banking services, with US$3.9 trillion in total assets, and ranked #1 in the Forbes Global 2000 ranking in 2023.""",
            "keyphrases": [
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
        }
    ],
)


TEXT_GENERATE_QUESTIONS = Prompt(
    name="text_generate_questions",
    instruction="Based on the given text and keyphrases, generate closed-ended questions that can be answered with '1' if the question can be answered using the text, or '0' if it cannot. The questions should ALWAYS result in a '1' based on the given text.",
    output_format_instruction=_output_instructions_question_generation,
    input_keys=["text", "keyphrases"],
    output_key="questions",
    output_type="json",
    examples=[
        {
            "text": """JPMorgan Chase & Co. is an American multinational finance company headquartered in New York City. It is the largest bank in the United States and the world's largest by market capitalization as of 2023. Founded in 1799, it is a major provider of investment banking services, with US$3.9 trillion in total assets, and ranked #1 in the Forbes Global 2000 ranking in 2023.""",
            "keyphrases": [
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
            "questions": [
                "Is JPMorgan Chase & Co. an American multinational finance company?",
                "Is JPMorgan Chase & Co. headquartered in New York City?",
                "Is JPMorgan Chase & Co. the largest bank in the United States?",
                "Is JPMorgan Chase & Co. the world's largest bank by market capitalization as of 2023?",
                "Was JPMorgan Chase & Co. founded in 1799?",
                "Is JPMorgan Chase & Co. a major provider of investment banking services?",
                "Does JPMorgan Chase & Co. have US$3.9 trillion in total assets?",
                "Was JPMorgan Chase & Co. ranked #1 in the Forbes Global 2000 ranking in 2023?",
            ],
        }
    ],
)


TEXT_GENERATE_ANSWERS = Prompt(
    name="text_generate_answers",
    instruction="Based on the list of close-ended '1' or '0' questions, generate a JSON with key 'answers', which is a list of strings that determines whether the provided summary contains sufficient information to answer EACH question. Answers should STRICTLY be either '1' or '0'. Answer '0' if the provided summary does not contain enough information to answer the question and answer '1' if the provided summary can answer the question.",
    output_format_instruction=_output_instructions_answer_generation,
    input_keys=["summary", "questions"],
    output_key="answers",
    output_type="json",
    examples=[
        {
            "summary": """JPMorgan Chase & Co., headquartered in New York City, is the largest bank in the US and the world's largest by market capitalization as of 2023. Founded in 1799, it offers extensive investment, private, asset management, and retail banking services, and has $3.9 trillion in assets, making it the fifth-largest bank globally. It operates the world's largest investment bank by revenue and was ranked #1 in the 2023 Forbes Global 2000.""",
            "questions": [
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
            "answers": ["0", "1", "1", "1", "0", "0", "1", "1", "1", "1", "1"],
        }
    ],
)


@dataclass
class SummarizationScore(MetricWithLLM):
    name: str = "summary_score"  # type: ignore
    max_retries: int = 1
    length_penalty: bool = True
    evaluation_mode: EvaluationMode = EvaluationMode.ca  # type: ignore[reportIncompatibleMethodOverride]
    question_generation_prompt: Prompt = field(
        default_factory=lambda: TEXT_GENERATE_QUESTIONS
    )
    answer_generation_prompt: Prompt = field(
        default_factory=lambda: TEXT_GENERATE_ANSWERS
    )

    def _get_extract_keyphrases_prompt(self, text) -> PromptValue:
        return TEXT_EXTRACT_KEYPHRASES.format(text=text)

    def _get_question_generation_prompt(self, text, keyphrases) -> PromptValue:
        return TEXT_GENERATE_QUESTIONS.format(text=text, keyphrases=keyphrases)

    def _get_answer_generation_prompt(
        self, questions: t.List, summary: str
    ) -> PromptValue:
        return TEXT_GENERATE_ANSWERS.format(summary=summary, questions=questions)

    async def _ascore(self, row: Dict, callbacks: Callbacks) -> float:
        # text is the contexts provided
        # summary is the summary generated by the model
        # TODO: add support for the query used as well
        text: str = "\n".join(row["contexts"])
        summary: str = row["summary"]
        keyphrases = await self._extract_keyphrases(text, callbacks)
        questions = await self._get_questions(text, keyphrases, callbacks)
        answers = await self._get_answers(questions, summary, callbacks)

        scores = []
        qa_score = self._compute_qa_score(answers)
        scores.append(qa_score)
        if self.length_penalty:
            conciseness_score = self._compute_conciseness_score(text, summary)
            scores.append(conciseness_score)
        return self._compute_score(scores)

    def _compute_score(self, scores) -> float:
        """Returns average score of the different scores."""
        return sum(scores) / len(scores)

    def _compute_qa_score(self, answers: t.List[str]) -> float:
        """Returns a score between 0 and 1 reflecting the fraction of
        correct answers, ie with a value 'yes'
        """
        correct = sum([1 for a in answers if a.lower() == "1"])
        return correct / len(answers)

    def _compute_conciseness_score(self, text, summary) -> float:
        """Returns the conciseness score of the summary. This is calculated as
        (1- relative_length_of_summary), where relative_length_of_summary is the
        ratio of the length of the summary to the length of the original text.
        This promotes shorter summaries.
        """
        return 1 - (len(summary) / len(text))

    async def _extract_keyphrases(self, text: str, callbacks: Callbacks) -> t.List[str]:
        assert self.llm is not None, "LLM is not initialized"
        p_value = self._get_extract_keyphrases_prompt(text)
        result = await self.llm.generate(
            prompt=p_value,
            callbacks=callbacks,
        )
        result_text = result.generations[0][0].text
        response = await _output_parser_keyphrase_extraction.aparse(
            result_text, p_value, self.llm, self.max_retries
        )

        if not response or not response.keyphrases:
            logging.error("No keyphrases generated, unable to calculate the score.")
            return []

        return response.keyphrases

    async def _get_questions(
        self, text: str, keyphrases: list[str], callbacks: Callbacks
    ) -> t.List[str]:
        assert self.llm is not None, "LLM is not initialized"
        p_value = self._get_question_generation_prompt(text, keyphrases)
        result = await self.llm.generate(
            prompt=p_value,
            callbacks=callbacks,
        )

        result_text = result.generations[0][0].text
        response = await _output_parser_question_generation.aparse(
            result_text, p_value, self.llm, self.max_retries
        )

        if not response or not response.questions:
            logging.error("No questions generated, unable to calculate the score.")
            return []

        return response.questions

    async def _get_answers(
        self, questions: t.List[str], summary: str, callbacks: Callbacks
    ) -> t.List[str]:
        assert self.llm is not None, "LLM is not initialized"
        p_value = self._get_answer_generation_prompt(questions, summary)
        result = await self.llm.generate(
            prompt=p_value,
            callbacks=callbacks,
        )

        result_text = result.generations[0][0].text
        response = await _output_parser_answer_generation.aparse(
            result_text, p_value, self.llm, self.max_retries
        )

        if not response or not response.answers:
            logger.error("No answers generated, unable to calculate the score.")
            return []

        return response.answers


def adapt(self, language: str, cache_dir: str | None = None) -> None:
    assert self.llm is not None, "set LLM before use"

    logger.info(f"Adapting summarization to {language}")
    self.question_generation_prompt = self.question_generation_prompt.adapt(
        language, self.llm, cache_dir
    )
    self.answer_generation_prompt = self.answer_generation_prompt.adapt(
        language, self.llm, cache_dir
    )


summarization_score = SummarizationScore()
