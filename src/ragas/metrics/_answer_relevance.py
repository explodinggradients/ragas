from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel, ValidationError
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithEmbeddings, MetricWithLLM
from ragas.llms.output_parser import get_json_format_instructions

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


class AnswerRelevanceClassification(BaseModel):
    question: str
    noncommittal: bool


_output_instructions = get_json_format_instructions(pydantic_object=AnswerRelevanceClassification)
_output_parser = PydanticOutputParser(pydantic_object=AnswerRelevanceClassification)


QUESTION_GEN = Prompt(
    name="question_generation",
    instruction="""Generate a question for the given answer and information context, and identify if the answer is noncommittal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers. Give noncommittal as "true" if the answer is noncommittal and "false" if the answer is committal.""",
    output_format_instruction=_output_instructions,
    examples=[
        {
            "answer": """Albert Einstein was born in Germany.""",
            "context": """Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time""",
            "output": AnswerRelevanceClassification.parse_obj({
                "question": "Where was Albert Einstein born?",
                "noncommittal": False,
            }).dict(),
        },
        {
            "answer": """It can change its skin color based on the temperature of its environment.""",
            "context": """A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment.""",
            "output": AnswerRelevanceClassification.parse_obj({
                "question": "What unique ability does the newly discovered species of frog have?",
                "noncommittal": False,
            }).dict(),
        },
        {
            "answer": """Everest""",
            "context": """The tallest mountain on Earth, measured from sea level, is a renowned peak located in the Himalayas.""",
            "output": AnswerRelevanceClassification.parse_obj({
                "question": "What is the tallest mountain on Earth?",
                "noncommittal": False,
            }).dict(),
        },
        {
            "answer": """I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. """,
            "context": """In 2023, a groundbreaking invention was announced: a smartphone with a battery life of one month, revolutionizing the way people use mobile technology.""",
            "output": AnswerRelevanceClassification.parse_obj({
                "question": "What was the groundbreaking feature of the smartphone invented in 2023?",
                "noncommittal": True,
            }).dict(),
        },
    ],
    input_keys=["answer", "context"],
    output_key="output",
    output_type="json",
)


@dataclass
class AnswerRelevancy(MetricWithLLM, MetricWithEmbeddings):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qac  # type: ignore
    question_generation: Prompt = field(default_factory=lambda: QUESTION_GEN)
    strictness: int = 3
    use_langchain_parser: bool = False

    def calculate_similarity(
        self: t.Self, question: str, generated_questions: list[str]
    ):
        assert self.embeddings is not None
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        ).reshape(len(generated_questions), -1)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

    def _calculate_score(self, answers: t.Sequence[AnswerRelevanceClassification], row: t.Dict) -> float:
        question = row["question"]
        gen_questions = [
            answer.question for answer in answers
        ]
        committal = np.any(
            [
                answer.noncommittal
                for answer in answers
            ]
        )
        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            score = np.nan
        else:
            cosine_sim = self.calculate_similarity(question, gen_questions)
            score = cosine_sim.mean() * int(not committal)

        return score

    def _create_question_gen_prompt(self, row: t.Dict) -> PromptValue:
        ans, ctx = row["answer"], row["contexts"]
        return self.question_generation.format(answer=ans, context="\n".join(ctx))

    async def _ascore(self, row: t.Dict, callbacks: Callbacks, is_async: bool) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt = self._create_question_gen_prompt(row)
        result = await self.llm.generate(
            prompt,
            n=self.strictness,
            callbacks=callbacks,
            is_async=is_async,
        )

        try:

            if self.use_langchain_parser:
                answers = [
                    _output_parser.parse(result.text)
                    for result in result.generations[0]
                ]
                # TODO: real error handling and retry?
                # https://python.langchain.com/docs/modules/model_io/output_parsers/types/retry
            else:
                responses = [
                    await json_loader.safe_load(r.text, self.llm, is_async=is_async)
                    for r in result.generations[0]
                ]
                answers = [
                    AnswerRelevanceClassification.parse_obj(response)
                    for response in responses
                ]
        except (OutputParserException, ValidationError) as err:
            print(f"Could not parse LLM response: {[result.text for result in result.generations[0]]}")
            print(f"Error: {err}")
            return np.nan

        return self._calculate_score(answers, row)

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logger.info(f"Adapting AnswerRelevancy metric to {language}")
        self.question_generation = self.question_generation.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.question_generation.save(cache_dir)


answer_relevancy = AnswerRelevancy()
