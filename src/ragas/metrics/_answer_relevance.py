from __future__ import annotations

import asyncio
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class ResponseRelevanceOutput(BaseModel):
    question: str
    noncommittal: int


class ResponseRelevanceInput(BaseModel):
    response: str


class ResponseRelevancePrompt(
    PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
):
    instruction = """Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers"""
    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    examples = [
        (
            ResponseRelevanceInput(
                response="""Albert Einstein was born in Germany.""",
            ),
            ResponseRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="""I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. """,
            ),
            ResponseRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]


@dataclass
class ResponseRelevancy(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
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

    name: str = "answer_relevancy"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    question_generation: PydanticPrompt = ResponseRelevancePrompt()
    strictness: int = 3

    def calculate_similarity(self, question: str, generated_questions: list[str]):
        assert self.embeddings is not None, (
            f"Error: '{self.name}' requires embeddings to be set."
        )
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)  # type: ignore[attr-defined]
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)  # type: ignore[attr-defined]
        ).reshape(len(generated_questions), -1)
        return self._cosine_similarity(question_vec, gen_question_vec)

    async def _acalculate_similarity(
        self, question: str, generated_questions: list[str]
    ) -> np.ndarray:
        assert self.embeddings is not None, (
            f"Error: '{self.name}' requires embeddings to be set."
        )

        async_embed_query = getattr(self.embeddings, "aembed_query", None)
        if async_embed_query is None:
            async_embed_query = getattr(self.embeddings, "aembed_text", None)
        if async_embed_query is not None:
            question_embedding = await async_embed_query(question)
        else:
            embed_query = getattr(self.embeddings, "embed_query", None)
            if embed_query is None:
                embed_query = getattr(self.embeddings, "embed_text")
            question_embedding = await asyncio.to_thread(embed_query, question)

        async_embed_documents = getattr(self.embeddings, "aembed_documents", None)
        if async_embed_documents is None:
            async_embed_documents = getattr(self.embeddings, "aembed_texts", None)
        if async_embed_documents is not None:
            generated_embeddings = await async_embed_documents(generated_questions)
        else:
            embed_documents = getattr(self.embeddings, "embed_documents", None)
            if embed_documents is None:
                embed_documents = getattr(self.embeddings, "embed_texts")
            generated_embeddings = await asyncio.to_thread(
                embed_documents, generated_questions
            )

        question_vec = np.asarray(question_embedding).reshape(1, -1)
        gen_question_vec = np.asarray(generated_embeddings).reshape(
            len(generated_questions), -1
        )
        return self._cosine_similarity(question_vec, gen_question_vec)

    @staticmethod
    def _cosine_similarity(
        question_vec: np.ndarray, gen_question_vec: np.ndarray
    ) -> np.ndarray:
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

    def _calculate_score(
        self, answers: t.Sequence[ResponseRelevanceOutput], row: t.Dict
    ) -> float:
        gen_questions = [answer.question for answer in answers]
        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            return np.nan

        cosine_sim = self.calculate_similarity(row["user_input"], gen_questions)
        return self._score_from_similarity(answers, cosine_sim)

    @staticmethod
    def _score_from_similarity(
        answers: t.Sequence[ResponseRelevanceOutput], cosine_sim: np.ndarray
    ) -> float:
        all_noncommittal = np.all([answer.noncommittal for answer in answers])
        return cosine_sim.mean() * int(not all_noncommittal)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_input = ResponseRelevanceInput(response=row["response"])

        responses = await self.question_generation.generate_multiple(
            data=prompt_input, llm=self.llm, callbacks=callbacks, n=self.strictness
        )

        gen_questions = [answer.question for answer in responses]
        if all(q == "" for q in gen_questions):
            return self._calculate_score(responses, row)

        cosine_sim = await self._acalculate_similarity(row["user_input"], gen_questions)
        return self._score_from_similarity(responses, cosine_sim)


class AnswerRelevancy(ResponseRelevancy):
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


answer_relevancy = AnswerRelevancy()
