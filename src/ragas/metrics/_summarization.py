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
    instruction="テキストを要約するために必要なキーフレーズを抽出してください。",
    output_format_instruction=_output_instructions_keyphrase_extraction,
    input_keys=["text"],
    output_key="keyphrases",
    output_type="json",
    examples=[
        {
            "text": """JPMorgan Chase & Co.は、ニューヨーク市に本社を置くアメリカの多国籍金融会社です。2023年時点で、アメリカ最大の銀行であり、時価総額で世界最大の銀行です。1799年に設立され、投資銀行業務の主要な提供者であり、総資産は3.9兆米ドルで、2023年のフォーブス・グローバル2000ランキングで第1位にランクされています。""",
            "keyphrases": [
                "JPMorgan Chase & Co.",
                "アメリカの多国籍金融会社",
                "ニューヨーク市に本社を置く",
                "アメリカ最大の銀行",
                "時価総額で世界最大の銀行",
                "1799年に設立",
                "投資銀行業務の主要な提供者",
                "総資産3.9兆米ドル",
                "フォーブス・グローバル2000ランキングで第1位にランク",
            ],
        }
    ],
)


TEXT_GENERATE_QUESTIONS = Prompt(
    name="text_generate_questions",
    instruction="与えられたテキストとキーフレーズに基づいて、テキストを使用して答えられる場合に '1' と回答し、答えられない場合に '0' と回答する閉じた質問を生成してください。質問は必ず与えられたテキストに基づいて '1' となるようにしてください。",
    output_format_instruction=_output_instructions_question_generation,
    input_keys=["text", "keyphrases"],
    output_key="questions",
    output_type="json",
    examples=[
        {
            "text": """JPMorgan Chase & Co.は、ニューヨーク市に本社を置くアメリカの多国籍金融会社です。2023年時点で、アメリカ最大の銀行であり、時価総額で世界最大の銀行です。1799年に設立され、投資銀行業務の主要な提供者であり、総資産は3.9兆米ドルで、2023年のフォーブス・グローバル2000ランキングで第1位にランクされています。""",
            "keyphrases": [
                "JPMorgan Chase & Co.",
                "アメリカの多国籍金融会社",
                "ニューヨーク市に本社を置く",
                "アメリカ最大の銀行",
                "時価総額で世界最大の銀行",
                "1799年に設立",
                "投資銀行業務の主要な提供者",
                "総資産3.9兆米ドル",
                "2023年のフォーブス・グローバル2000ランキングで第1位にランク",
            ],
            "questions": [
                "JPMorgan Chase & Co.はアメリカの多国籍金融会社ですか？",
                "JPMorgan Chase & Co.はニューヨーク市に本社がありますか？",
                "JPMorgan Chase & Co.はアメリカ最大の銀行ですか？",
                "JPMorgan Chase & Co.は2023年時点で時価総額で世界最大の銀行ですか？",
                "JPMorgan Chase & Co.は1799年に設立されましたか？",
                "JPMorgan Chase & Co.は投資銀行業務の主要な提供者ですか？",
                "JPMorgan Chase & Co.の総資産は3.9兆米ドルですか？",
                "JPMorgan Chase & Co.は2023年のフォーブス・グローバル2000ランキングで第1位にランクされましたか？",
            ],
        }
    ],
)


TEXT_GENERATE_ANSWERS = Prompt(
    name="text_generate_answers",
    instruction="与えられた '1' または '0' で回答する閉じた質問のリストに基づいて、キーが 'answers' であるJSONを生成してください。このキーには、各質問に対して提供された要約が十分な情報を含んでいるかどうかを決定する文字列のリストが含まれます。回答は厳密に '1' または '0' のみとし、提供された要約が質問に答えるための十分な情報を含んでいない場合は '0'、答えられる場合は '1' としてください。",
    output_format_instruction=_output_instructions_answer_generation,
    input_keys=["summary", "questions"],
    output_key="answers",
    output_type="json",
    examples=[
        {
            "summary": """JPMorgan Chase & Co.はニューヨーク市に本社を置き、2023年時点で米国最大の銀行であり、時価総額で世界最大の銀行です。1799年に設立され、広範な投資、プライベート、資産管理、およびリテールバンキングサービスを提供しており、総資産は3.9兆ドルで、世界で5番目に大きい銀行です。同社は収益で世界最大の投資銀行を運営しており、2023年のフォーブス・グローバル2000ランキングで第1位にランクされました。""",
            "questions": [
                "JPMorgan Chase & Co.はアメリカの多国籍金融会社ですか？",
                "JPMorgan Chase & Co.はニューヨーク市に本社がありますか？",
                "JPMorgan Chase & Co.はアメリカ最大の銀行ですか？",
                "JPMorgan Chase & Co.は2023年時点で時価総額で世界最大の銀行ですか？",
                "JPMorgan Chase & Co.は金融安定理事会によってシステム上重要と見なされていますか？",
                "JPMorgan Chase & Co.は1799年にチェース・マンハッタン・カンパニーとして設立されましたか？",
                "JPMorgan Chase & Co.は投資銀行業務の主要な提供者ですか？",
                "JPMorgan Chase & Co.は世界で資産規模で5番目に大きい銀行ですか？",
                "JPMorgan Chase & Co.は収益で最大の投資銀行を運営していますか？",
                "JPMorgan Chase & Co.はフォーブス・グローバル2000ランキングで第1位にランクされましたか？",
                "JPMorgan Chase & Co.は投資銀行業務を提供していますか？",
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
