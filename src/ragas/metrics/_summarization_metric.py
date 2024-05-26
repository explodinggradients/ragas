from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from typing import Dict

import json
import numpy as np
from langchain.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt, PromptValue
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

logger = logging.getLogger(__name__)

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

_output_parser_question_generation = RagasoutputParser(pydantic_object=GenerateQuestionsResponse)
_output_parser_answer_generation = RagasoutputParser(pydantic_object=GenerateAnswersResponse)

GENERATE_QUESTION_INSTRUCTION = """\
Based on the given text, generate "n" closed-ended questions that can be answered with either a 'yes' or 'no'. The questions generated should ALWAYS result in a 'yes' based on the given text.    
** IMPORTANT
Only return a JSON with a 'questions' key, which is a list of strings. The questions have to be STRICTLY closed ended. The given text should be able to answer 'yes' for each question.
**
"""
TEXT_GENERATE_QUESTIONS = Prompt(
    name="text_generate_questions",
    instruction=GENERATE_QUESTION_INSTRUCTION,
    output_format_instruction=_output_instructions_question_generation,
    input_keys=["text", "n"],
    output_key="questions",
    output_type="json",
    examples=[
        {
            "text": """JPMorgan Chase & Co. is an American multinational finance company headquartered in New York City and incorporated in Delaware. It is the largest bank in the United States and the world's largest bank by market capitalization as of 2023. As the largest of Big Four banks, the firm is considered systemically important by the Financial Stability Board. Its size and scale have often led to enhanced regulatory oversight as well as the maintenance of an internal "Fortress Balance Sheet". The firm is headquartered at 383 Madison Avenue in Midtown Manhattan and is set to move into the under-construction JPMorgan Chase Building at 270 Park Avenue in 2025.
                    The firm's early history can be traced to 1799, with the founding of what became the Chase Manhattan Company. In 1871, J.P. Morgan & Co. was founded by J. P. Morgan who launched the House of Morgan on 23 Wall Street as a national purveyor of commercial, investment, and private banking services. The present company was formed after the two predecessor firms merged in 2000, creating a diversified holding entity. It is a major provider of investment banking services, through corporate advisory, mergers and acquisitions, sales and trading, and public offerings. Their private banking franchise and asset management division are among the world's largest in terms of total assets. Its retail banking and credit card offerings are provided via the Chase brand in the U.S. and United Kingdom.
                    With US$3.9 trillion in total assets, JPMorgan Chase is the fifth-largest bank in the world by assets. The firm operates the largest investment bank in the world by revenue. It occupies 24th place on the Fortune 500 list of the largest U.S. corporations by revenue. In 2023, JPMorgan Chase was ranked #1 in the Forbes Global 2000 ranking. It receives routine criticism for its risk management, broad financing activities, and large-scale legal settlements.
                    """,
            "n": 5,
            "questions": [
                "Is JPMorgan Chase & Co. an American multinational finance company?",
                "Is JPMorgan Chase the largest bank in the United States?",
                "Is JPMorgan Chase the world's largest bank by market capitalization as of 2023?",
                "Is JPMorgan Chase considered systemically important by the Financial Stability Board?",
                "Is JPMorgan Chase headquartered in New York City?"
            ]
        },
        {
            "text": """Photosynthesis, the process by which green plants and certain other organisms transform light energy into chemical energy. During photosynthesis in green plants, light energy is captured and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds.
                It would be impossible to overestimate the importance of photosynthesis in the maintenance of life on Earth. If photosynthesis ceased, there would soon be little food or other organic matter on Earth. Most organisms would disappear, and in time Earth’s atmosphere would become nearly devoid of gaseous oxygen. 
                The only organisms able to exist under such conditions would be the chemosynthetic bacteria, which can utilize the chemical energy of certain inorganic compounds and thus are not dependent on the conversion of light energy.
                Energy produced by photosynthesis carried out by plants millions of years ago is responsible for the fossil fuels (i.e., coal, oil, and gas) that power industrial society. In past ages, green plants and small organisms that fed on plants increased faster than they were consumed, and their remains were deposited in Earth’s crust by sedimentation and other geological processes. There, protected from oxidation, these organic remains were slowly converted to fossil fuels. These fuels not only provide much of the energy used in factories, homes, and transportation but also serve as the raw material for plastics and other synthetic products. Unfortunately, modern civilization is using up in a few centuries the excess of photosynthetic production accumulated over millions of years. Consequently, the carbon dioxide that has been removed from the air to make carbohydrates in photosynthesis over millions of years is being returned at an incredibly rapid rate. The carbon dioxide concentration in Earth’s atmosphere is rising the fastest it ever has in Earth’s history, and this phenomenon is expected to have major implications on Earth’s climate.
                """,
            "n": 7,
            "questions": [
                "Is photosynthesis the process by which green plants and certain other organisms transform light energy into chemical energy?",
                "Would there be little food or other organic matter on Earth if photosynthesis ceased?",
                "Would most organisms disappear if photosynthesis ceased?",
                "Would Earth’s atmosphere become nearly devoid of gaseous oxygen if photosynthesis ceased?",
                "Are the only organisms able to exist under conditions without photosynthesis the chemosynthetic bacteria?",
                "Is the energy produced by photosynthesis responsible for fossil fuels that power industrial society?",
                "Is the carbon dioxide concentration in Earth’s atmosphere rising at the fastest rate in Earth’s history?"
            ]
        }
    ]
)


GENERATE_ANSWER_INSTRUCTION = """\
Based on the list of close-ended 'yes' or 'no' questions, generate a JSON with key 'answers', which is a list of strings that determines whether the provided summary contains sufficient information to answer EACH question. Answers should STRICTLY be either 'yes' or 'no'. Answer 'no' if the provided summary does not contain enough information to answer the question.
** IMPORTANT 
Please make sure to ONLY return in JSON format, with the 'answers' key as a list of strings. Only return the JSON with the 'answers' key. The length of 'answers' SHOULD BE STRICTLY EQUAL to that of questions.
**
"""
TEXT_GENERATE_ANSWERS = Prompt(
    name="text_generate_answers",
    instruction=GENERATE_ANSWER_INSTRUCTION,
    output_format_instruction=_output_instructions_answer_generation,
    input_keys=["summary", "questions"],
    output_key="answers",
    output_type="json",
    examples=[
        {
            "summary": """JPMorgan Chase & Co., headquartered in New York City and incorporated in Delaware, is the largest bank in the US and the world's largest by market capitalization as of 2023. It is systemically important by the Financial Stability Board, leading to enhanced regulatory oversight and maintaining a 'Fortress Balance Sheet'. The firm, originally founded in 1799, was formed in its current state through a merger in 2000. It offers extensive investment, private, asset management, and retail banking services, and has $3.9 trillion in assets, making it the fifth-largest bank globally. JPMorgan Chase also operates the world's largest investment bank by revenue, ranks 24th on the Fortune 500 list, and was ranked #1 in the 2023 Forbes Global 2000. It faces criticism for risk management and legal issues.""",
            "questions": [
                "Is JPMorgan Chase & Co. an American multinational finance company?",
                "Is JPMorgan Chase the largest bank in the United States?",
                "Is JPMorgan Chase the world's largest bank by market capitalization as of 2023?",
                "Is JPMorgan Chase considered systemically important by the Financial Stability Board?",
                "Is JPMorgan Chase headquartered in New York City?"
            ],
            "answers": ["yes", "yes", "yes", "yes", "yes"]
        },
        {
            "summary": """Photosynthesis is the process by which green plants and certain organisms convert light energy into chemical energy, producing oxygen and organic compounds. Its importance is crucial as its cessation would lead to a lack of food and organic matter, and a near-absence of atmospheric oxygen. Photosynthesis from past ages has resulted in fossil fuels, which power modern society, but their rapid consumption is causing a rapid increase in carbon dioxide levels, with significant implications for Earth's climate.""",
            "questions": [
                "Is photosynthesis the process by which green plants and certain other organisms transform light energy into chemical energy?",
                "Would there be little food or other organic matter on Earth if photosynthesis ceased?",
                "Would most organisms disappear if photosynthesis ceased?",
                "Would Earth’s atmosphere become nearly devoid of gaseous oxygen if photosynthesis ceased?",
                "Are the only organisms able to exist under conditions without photosynthesis the chemosynthetic bacteria?",
                "Is the energy produced by photosynthesis responsible for fossil fuels that power industrial society?",
                "Is the carbon dioxide concentration in Earth’s atmosphere rising at the fastest rate in Earth’s history?"
            ],
            "answers": ["yes", "yes", "no", "yes", "no", "yes", "yes"]
        }
    ]
)

@dataclass
class SummarizationMetric(MetricWithLLM):
    """Given a text and its generated summary, calculates a score for 
    quantifying the quality of the summary. Currently we use the following method
    to quantify it:
    - Given the original text, generate a set of 'yes'/'no' questions based on it.
    - Ask those questions to the generated summary and find out how many of them
    are answered correctly.
    - Return the fraction of correctly answered questions as the score.
    
    Also, this metric can be configured to include a length penalty for the summary,
    thereby not being biased towards summaries that are same as the original text. 
    If this option is configured, this metric will also consider the conciseness of
    the summary, and calculate the score as the average of the QA score and the
    conciseness score.
    """

    name: str = "summary_score" # type: ignore
    n_questions: int = 5
    max_retries: int = 1
    length_penalty: bool = True
    evaluation_mode: EvaluationMode = EvaluationMode.ts # type: ignore
    question_generation_prompt: Prompt = field(default_factory=lambda: TEXT_GENERATE_QUESTIONS)
    answer_generation_prompt: Prompt = field(default_factory=lambda: TEXT_GENERATE_ANSWERS)

    
    def _get_question_generation_prompt(self, text) -> PromptValue:
        return TEXT_GENERATE_QUESTIONS.format(text=text, n=self.n_questions)
    
    def _get_answer_generation_prompt(self, questions: t.List, summary: str) -> PromptValue:
        return TEXT_GENERATE_ANSWERS.format(summary=summary, questions=questions)
    
    async def _ascore(self, row: Dict, callbacks: Callbacks, is_async: bool) -> float:
        text, summary = row["text"], row["summary"]
        questions = await self._get_questions(text, callbacks, is_async)
        answers = await self._get_answers(questions, summary, callbacks, is_async)

        qa_score = self._compute_qa_score(answers)
        conciseness_score = self._compute_conciseness_score(text, summary)
        return self._compute_score(qa_score, conciseness_score)
    
    def _compute_score(self, qa_score, conciseness_score) -> float:
        """Returns the QA score if length_penalty is False else
        returns the averaged score of the QA and conciseness scores.
        """
        if self.length_penalty:
            return (qa_score + conciseness_score)/2
        return qa_score
    
    def _compute_qa_score(self, answers: t.List) -> float:
        """Returns a score between 0 and 1 reflecting the fraction of
        correct answers, ie with a value 'yes'
        """
        correct = sum([1 for a in answers if a.lower() == "yes"])
        return correct / len(answers)
    
    def _compute_conciseness_score(self, text, summary) -> float:
        """Returns the conciseness score of the summary. This is calculated as 
        (1- relative_length_of_summary), where relative_length_of_summary is the
        ratio of the length of the summary to the length of the original text.
        This promotes shorter summaries.
        """
        return 1-(len(summary) / len(text))
    
    async def _get_questions(self, text: str, callbacks: Callbacks, is_async: bool) -> t.List[str]:
        assert self.llm is not None, "LLM is not initialized"
        p_value = self._get_question_generation_prompt(text)
        result = await self.llm.generate(
            prompt=p_value,
            callbacks=callbacks,
            is_async=is_async,
        )

        result_text = result.generations[0][0].text
        answer = await _output_parser_question_generation.aparse(
            result_text, p_value, self.llm, self.max_retries
        )
        if answer is None:
            return []
        
        return answer.questions
    
    async def _get_answers(self, questions: t.List[str], summary: str, callbacks: Callbacks, is_async: bool) -> t.List[str]:
        assert self.llm is not None, "LLM is not initialized"
        p_value = self._get_answer_generation_prompt(questions, summary)
        result = await self.llm.generate(
            prompt=p_value,
            callbacks=callbacks,
            is_async=is_async,
        )

        result_text = result.generations[0][0].text
        answer = await _output_parser_answer_generation.aparse(
            result_text, p_value, self.llm, self.max_retries
        )
        if answer is None:
            return []
        
        return answer.answers

summary_score = SummarizationMetric()