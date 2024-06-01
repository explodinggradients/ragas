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
_output_parser_question_generation = RagasoutputParser(pydantic_object=GenerateQuestionsResponse)
_output_parser_answer_generation = RagasoutputParser(pydantic_object=GenerateAnswersResponse)
_output_parser_keyphrase_extraction = RagasoutputParser(pydantic_object=ExtractKeyphrasesResponse)

EXTRACT_KEYPHRASE_INSTRUCTION = """\
Extract the most crucial keyphrases that are essential for summarizing the text.

** Important
Return in JSON format with 'keyphrases' as key and a list of the extracted items as the value. Only return the JSON.
"""
TEXT_EXTRACT_KEYPHRASES = Prompt(
    name="text_extract_keyphrases",
    instruction=EXTRACT_KEYPHRASE_INSTRUCTION,
    output_format_instruction=_output_instructions_keyphrase_extraction,
    input_keys=["text"],
    output_key="keyphrases",
    output_type="json",
    examples=[
        {
            "text": """JPMorgan Chase & Co. is an American multinational finance company headquartered in New York City and incorporated in Delaware. It is the largest bank in the United States and the world's largest bank by market capitalization as of 2023. As the largest of Big Four banks, the firm is considered systemically important by the Financial Stability Board. Its size and scale have often led to enhanced regulatory oversight as well as the maintenance of an internal "Fortress Balance Sheet". The firm is headquartered at 383 Madison Avenue in Midtown Manhattan and is set to move into the under-construction JPMorgan Chase Building at 270 Park Avenue in 2025.
                    The firm's early history can be traced to 1799, with the founding of what became the Chase Manhattan Company. In 1871, J.P. Morgan & Co. was founded by J. P. Morgan who launched the House of Morgan on 23 Wall Street as a national purveyor of commercial, investment, and private banking services. The present company was formed after the two predecessor firms merged in 2000, creating a diversified holding entity. It is a major provider of investment banking services, through corporate advisory, mergers and acquisitions, sales and trading, and public offerings. Their private banking franchise and asset management division are among the world's largest in terms of total assets. Its retail banking and credit card offerings are provided via the Chase brand in the U.S. and United Kingdom.
                    With US$3.9 trillion in total assets, JPMorgan Chase is the fifth-largest bank in the world by assets. The firm operates the largest investment bank in the world by revenue. It occupies 24th place on the Fortune 500 list of the largest U.S. corporations by revenue. In 2023, JPMorgan Chase was ranked #1 in the Forbes Global 2000 ranking. It receives routine criticism for its risk management, broad financing activities, and large-scale legal settlements.
                    """,
            "keyphrases": [
                "JPMorgan Chase & Co.",
                "American multinational finance company",
                "headquartered in New York City",
                "largest bank in the United States",
                "world's largest bank by market capitalization",
                "systemically important by the Financial Stability Board",
                "founded in 1799 as Chase Manhattan Company",
                "Major provider of investment banking services",
                "Fifth-largest bank in the world by assets",
                "Operates the largest investment bank by revenue",
                "Ranked #1 in Forbes Global 2000 ranking",
                "investment banking services",
                ]
        },
        {
            "text": """Photosynthesis, the process by which green plants and certain other organisms transform light energy into chemical energy. During photosynthesis in green plants, light energy is captured and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds.
                    It would be impossible to overestimate the importance of photosynthesis in the maintenance of life on Earth. If photosynthesis ceased, there would soon be little food or other organic matter on Earth. Most organisms would disappear, and in time Earth’s atmosphere would become nearly devoid of gaseous oxygen. 
                    The only organisms able to exist under such conditions would be the chemosynthetic bacteria, which can utilize the chemical energy of certain inorganic compounds and thus are not dependent on the conversion of light energy.
                    Energy produced by photosynthesis carried out by plants millions of years ago is responsible for the fossil fuels (i.e., coal, oil, and gas) that power industrial society. In past ages, green plants and small organisms that fed on plants increased faster than they were consumed, and their remains were deposited in Earth’s crust by sedimentation and other geological processes. There, protected from oxidation, these organic remains were slowly converted to fossil fuels. These fuels not only provide much of the energy used in factories, homes, and transportation but also serve as the raw material for plastics and other synthetic products. Unfortunately, modern civilization is using up in a few centuries the excess of photosynthetic production accumulated over millions of years. Consequently, the carbon dioxide that has been removed from the air to make carbohydrates in photosynthesis over millions of years is being returned at an incredibly rapid rate. The carbon dioxide concentration in Earth’s atmosphere is rising the fastest it ever has in Earth’s history, and this phenomenon is expected to have major implications on Earth’s climate.
                    """,
            "keyphrases": [
                "Photosynthesis",
                "Green plants",
                "Transform light energy into chemical energy",
                "Importance of photosynthesis in maintaining life on Earth",
                "Impact of ceasing photosynthesis",
                "Chemosynthetic bacteria",
                "Energy produced by photosynthesis",
                "Fossil fuels",
                "Carbon dioxide concentration in Earth's atmosphere"
            ]
        }
    ]
)



GENERATE_QUESTION_INSTRUCTION = """\
Based on the given text and a list of keyphrases, generate closed-ended questions that can be answered with either a '1' if the question can be answered using the text, or '0' if it cannot be answered using the text. The questions generated should ALWAYS result in a '1' based on the given text.    
** IMPORTANT
Only return a JSON with a 'questions' key and the list of questions as value. The questions have to be STRICTLY closed ended. The given text should be able to answer '1' for each question.
"""
TEXT_GENERATE_QUESTIONS = Prompt(
    name="text_generate_questions",
    instruction=GENERATE_QUESTION_INSTRUCTION,
    output_format_instruction=_output_instructions_question_generation,
    input_keys=["text", "keyphrases"],
    output_key="questions",
    output_type="json",
    examples=[
        {
            "text": """JPMorgan Chase & Co. is an American multinational finance company headquartered in New York City and incorporated in Delaware. It is the largest bank in the United States and the world's largest bank by market capitalization as of 2023. As the largest of Big Four banks, the firm is considered systemically important by the Financial Stability Board. Its size and scale have often led to enhanced regulatory oversight as well as the maintenance of an internal "Fortress Balance Sheet". The firm is headquartered at 383 Madison Avenue in Midtown Manhattan and is set to move into the under-construction JPMorgan Chase Building at 270 Park Avenue in 2025.
                    The firm's early history can be traced to 1799, with the founding of what became the Chase Manhattan Company. In 1871, J.P. Morgan & Co. was founded by J. P. Morgan who launched the House of Morgan on 23 Wall Street as a national purveyor of commercial, investment, and private banking services. The present company was formed after the two predecessor firms merged in 2000, creating a diversified holding entity. It is a major provider of investment banking services, through corporate advisory, mergers and acquisitions, sales and trading, and public offerings. Their private banking franchise and asset management division are among the world's largest in terms of total assets. Its retail banking and credit card offerings are provided via the Chase brand in the U.S. and United Kingdom.
                    With US$3.9 trillion in total assets, JPMorgan Chase is the fifth-largest bank in the world by assets. The firm operates the largest investment bank in the world by revenue. It occupies 24th place on the Fortune 500 list of the largest U.S. corporations by revenue. In 2023, JPMorgan Chase was ranked #1 in the Forbes Global 2000 ranking. It receives routine criticism for its risk management, broad financing activities, and large-scale legal settlements.
                    """,
            "keyphrases": [
                "JPMorgan Chase & Co.",
                "American multinational finance company",
                "headquartered in New York City",
                "largest bank in the United States",
                "world's largest bank by market capitalization",
                "systemically important by the Financial Stability Board",
                "founded in 1799 as Chase Manhattan Company",
                "Major provider of investment banking services",
                "Fifth-largest bank in the world by assets",
                "Operates the largest investment bank by revenue",
                "Ranked #1 in Forbes Global 2000 ranking",
                "investment banking services",
                ],
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
                "Does JPMorgan Chase & Co. provide investment banking services?"
            ]
        },
        {
            "text": """Photosynthesis, the process by which green plants and certain other organisms transform light energy into chemical energy. During photosynthesis in green plants, light energy is captured and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds.
                It would be impossible to overestimate the importance of photosynthesis in the maintenance of life on Earth. If photosynthesis ceased, there would soon be little food or other organic matter on Earth. Most organisms would disappear, and in time Earth’s atmosphere would become nearly devoid of gaseous oxygen. 
                The only organisms able to exist under such conditions would be the chemosynthetic bacteria, which can utilize the chemical energy of certain inorganic compounds and thus are not dependent on the conversion of light energy.
                Energy produced by photosynthesis carried out by plants millions of years ago is responsible for the fossil fuels (i.e., coal, oil, and gas) that power industrial society. In past ages, green plants and small organisms that fed on plants increased faster than they were consumed, and their remains were deposited in Earth’s crust by sedimentation and other geological processes. There, protected from oxidation, these organic remains were slowly converted to fossil fuels. These fuels not only provide much of the energy used in factories, homes, and transportation but also serve as the raw material for plastics and other synthetic products. Unfortunately, modern civilization is using up in a few centuries the excess of photosynthetic production accumulated over millions of years. Consequently, the carbon dioxide that has been removed from the air to make carbohydrates in photosynthesis over millions of years is being returned at an incredibly rapid rate. The carbon dioxide concentration in Earth’s atmosphere is rising the fastest it ever has in Earth’s history, and this phenomenon is expected to have major implications on Earth’s climate.
                """,
            "keyphrases": [
                "Photosynthesis",
                "Green plants",
                "Transform light energy into chemical energy",
                "Importance of photosynthesis in maintaining life on Earth",
                "Impact of ceasing photosynthesis",
                "Chemosynthetic bacteria",
                "Energy produced by photosynthesis",
                "Fossil fuels",
                "Carbon dioxide concentration in Earth's atmosphere"
            ],
            "questions": [
                "Is photosynthesis the process by which green plants transform light energy into chemical energy?",
                "Can photosynthesis be overestimated in terms of importance in maintaining life on Earth?",
                "What would happen if photosynthesis ceased?",
                "Are chemosynthetic bacteria able to exist under conditions where photosynthesis has ceased?",
                "Is energy produced by photosynthesis responsible for fossil fuels?",
                "Does photosynthesis impact the carbon dioxide concentration in Earth's atmosphere?",
                "Can green plants transform light energy into chemical energy?",
                "Is the importance of photosynthesis in maintaining life on Earth high?",
                "Are fossil fuels produced by the energy from photosynthesis?",
                "Is the carbon dioxide concentration in Earth's atmosphere rising due to photosynthesis?"
            ]
        }
    ]
)


GENERATE_ANSWER_INSTRUCTION = """\
Based on the list of close-ended '1' or '0' questions, generate a JSON with key 'answers', which is a list of strings that determines whether the provided summary contains sufficient information to answer EACH question. Answers should STRICTLY be either '1' or '0'. Answer '0' if the provided summary does not contain enough information to answer the question and answer '1' if the provided summary can answer the question.
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
                "Is JPMorgan Chase & Co. headquartered in New York City?",
                "Is JPMorgan Chase & Co. the largest bank in the United States?",
                "Is JPMorgan Chase & Co. the world's largest bank by market capitalization as of 2023?",
                "Is JPMorgan Chase & Co. considered systemically important by the Financial Stability Board?",
                "Was JPMorgan Chase & Co. founded in 1799 as the Chase Manhattan Company?",
                "Is JPMorgan Chase & Co. a major provider of investment banking services?",
                "Is JPMorgan Chase & Co. the fifth-largest bank in the world by assets?",
                "Does JPMorgan Chase & Co. operate the largest investment bank by revenue?",
                "Was JPMorgan Chase & Co. ranked #1 in the Forbes Global 2000 ranking?",
                "Does JPMorgan Chase & Co. provide investment banking services?"
            ],
            "answers": [
                "1",
                "1",
                "1",
                "1",
                "1",
                "0",
                "1",
                "1",
                "1",
                "1",
                "1"
            ]
        },
        {
            "summary": """Photosynthesis is the process by which green plants and certain organisms convert light energy into chemical energy, producing oxygen and organic compounds from water, carbon dioxide, and minerals. It is crucial for life on Earth, as its absence would result in a lack of food and oxygen, causing most organisms to disappear. Only chemosynthetic bacteria, which use chemical energy from inorganic compounds, would survive. Photosynthesis from millions of years ago created fossil fuels, essential for modern society, but their rapid consumption is increasing carbon dioxide levels at an unprecedented rate, significantly impacting Earth's climate.""",
            "questions": [
                "Is photosynthesis the process by which green plants transform light energy into chemical energy?",
                "Can photosynthesis be overestimated in terms of importance in maintaining life on Earth?",
                "What would happen if photosynthesis ceased?",
                "Are chemosynthetic bacteria able to exist under conditions where photosynthesis has ceased?",
                "Is energy produced by photosynthesis responsible for fossil fuels?",
                "Does photosynthesis impact the carbon dioxide concentration in Earth's atmosphere?",
                "Can green plants transform light energy into chemical energy?",
                "Is the importance of photosynthesis in maintaining life on Earth high?",
                "Are fossil fuels produced by the energy from photosynthesis?",
                "Is the carbon dioxide concentration in Earth's atmosphere rising due to photosynthesis?"
            ],
            "answers": [
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1"
            ]
        }
    ]
)

@dataclass
class SummarizationMetric(MetricWithLLM):
    """Given a text and its generated summary, calculates a score for 
    quantifying the quality of the summary. Currently we use the following method
    to quantify it:
    - Given the original text, extract a set of entities, events, or concepts, which
    we call keyphrases, from it
    - Generate a set of 'yes'/'no'(1/0) questions based on the text and the extracted keyphrases.
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
    max_retries: int = 1
    length_penalty: bool = True
    evaluation_mode: EvaluationMode = EvaluationMode.ts # type: ignore
    question_generation_prompt: Prompt = field(default_factory=lambda: TEXT_GENERATE_QUESTIONS)
    answer_generation_prompt: Prompt = field(default_factory=lambda: TEXT_GENERATE_ANSWERS)

    def _get_extract_keyphrases_prompt(self, text) -> PromptValue:
        return TEXT_EXTRACT_KEYPHRASES.format(text=text)
        
    def _get_question_generation_prompt(self, text, keyphrases) -> PromptValue:
        return TEXT_GENERATE_QUESTIONS.format(text=text, keyphrases=keyphrases)
    
    def _get_answer_generation_prompt(self, questions: t.List, summary: str) -> PromptValue:
        return TEXT_GENERATE_ANSWERS.format(summary=summary, questions=questions)
    
    async def _ascore(self, row: Dict, callbacks: Callbacks, is_async: bool) -> float:
        text, summary = row["text"], row["summary"]
        keyphrases = await self._extract_keyphrases(text, callbacks, is_async)
        questions = await self._get_questions(text, keyphrases, callbacks, is_async)
        answers = await self._get_answers(questions, summary, callbacks, is_async)

        scores = []
        qa_score = self._compute_qa_score(answers)
        scores.append(qa_score)
        if self.length_penalty:
            conciseness_score = self._compute_conciseness_score(text, summary)
            scores.append(conciseness_score)
        return self._compute_score(scores)
    
    def _compute_score(self, scores) -> float:
        """Returns average score of the different scores.
        """
        return sum(scores)/len(scores)
    
    def _compute_qa_score(self, answers: t.List) -> float:
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
        return 1-(len(summary) / len(text))
    
    
    async def _extract_keyphrases(self, text: str, callbacks: Callbacks, is_async: bool) -> t.List[str]:
        assert self.llm is not None, "LLM is not initialized"
        p_value = self._get_extract_keyphrases_prompt(text)
        result = await self.llm.generate(
            prompt=p_value,
            callbacks=callbacks,
            is_async=is_async,
        )
        result_text = result.generations[0][0].text
        answer = await _output_parser_keyphrase_extraction.aparse(
            result_text, p_value, self.llm, self.max_retries
        )
        return answer.keyphrases
    
    async def _get_questions(self, text: str, keyphrases: list[str], callbacks: Callbacks, is_async: bool) -> t.List[str]:
        assert self.llm is not None, "LLM is not initialized"
        p_value = self._get_question_generation_prompt(text, keyphrases)
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