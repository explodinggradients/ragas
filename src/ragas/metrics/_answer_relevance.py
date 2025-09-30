from __future__ import annotations

import logging
import re
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
from ragas.metrics.prompts import ANSWER_RELEVANCY_DIRECT_SCORING
from ragas.prompt import Prompt, PydanticPrompt

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


# Modern implementation using simple prompts
class ModernAnswerRelevancyImpl:
    """Internal modern implementation using simple string prompts."""

    def __init__(self):
        self.prompt = Prompt(ANSWER_RELEVANCY_DIRECT_SCORING)

    async def ascore(
        self, question: str, answer: str, llm: t.Any
    ) -> t.Dict[str, t.Any]:
        """Score answer relevancy using simple prompt and text parsing."""

        # Format the prompt
        formatted_prompt = self.prompt.format(question=question, answer=answer)

        # Get response from LLM (as plain text)
        if hasattr(llm, "agenerate"):
            # Modern LLM with structured output capability
            try:
                # Try to get structured output first
                from pydantic import BaseModel

                class SimpleRelevanceOutput(BaseModel):
                    score: float
                    reasoning: str
                    is_noncommittal: bool

                response = await llm.agenerate(formatted_prompt, SimpleRelevanceOutput)
                return {
                    "score": response.score,
                    "reasoning": response.reasoning,
                    "is_noncommittal": response.is_noncommittal,
                }
            except Exception:
                # Fallback to text generation and parsing
                pass

        # Fallback: Use text generation and parse the response
        if hasattr(llm, "agenerate_text"):
            from langchain_core.prompt_values import StringPromptValue

            llm_result = await llm.agenerate_text(
                StringPromptValue(text=formatted_prompt)
            )
            response_text = llm_result.generations[0][0].text
        else:
            # Last resort - assume it's a simple text generator
            response_text = await llm.agenerate(formatted_prompt)
            if hasattr(response_text, "text"):
                response_text = response_text.text
            elif not isinstance(response_text, str):
                response_text = str(response_text)

        # Parse the response text
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> t.Dict[str, t.Any]:
        """Parse the LLM response text to extract score, reasoning, and noncommittal flag."""

        # Default values
        score = 0.0
        reasoning = "Could not parse response"
        is_noncommittal = False

        try:
            # Extract score using regex
            score_match = re.search(
                r"Score:\s*([0-9]*\.?[0-9]+)", response_text, re.IGNORECASE
            )
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            # Extract reasoning
            reasoning_match = re.search(
                r"Reasoning:\s*(.+?)(?=\n|Is_Noncommittal:|$)",
                response_text,
                re.IGNORECASE | re.DOTALL,
            )
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()

            # Extract noncommittal flag
            noncommittal_match = re.search(
                r"Is_Noncommittal:\s*(true|false)", response_text, re.IGNORECASE
            )
            if noncommittal_match:
                is_noncommittal = noncommittal_match.group(1).lower() == "true"

        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
            logger.warning(f"Response text: {response_text[:200]}...")

        return {
            "score": score,
            "reasoning": reasoning,
            "is_noncommittal": is_noncommittal,
        }


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
        question = row["user_input"]
        gen_questions = [answer.question for answer in answers]
        all_noncommittal = np.all([answer.noncommittal for answer in answers])
        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            score = np.nan
        else:
            cosine_sim = self.calculate_similarity(question, gen_questions)
            score = cosine_sim.mean() * int(not all_noncommittal)

        return score

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

        return self._calculate_score(responses, row)


class AnswerRelevancy(ResponseRelevancy):
    """
    Smart answer relevancy that automatically routes to appropriate implementation.

    - Legacy LLMs → Uses original question generation + embeddings approach
    - Modern LLMs → Uses SimpleLLMMetric-based direct scoring approach
    - Same interface for both → Callers don't know the difference
    """

    # Modern implementation components (lazy-loaded)
    _modern_metric: t.Optional[ModernAnswerRelevancyImpl] = None

    def _get_modern_metric(self) -> ModernAnswerRelevancyImpl:
        """Lazy-load modern metric implementation."""
        if self._modern_metric is None:
            self._modern_metric = ModernAnswerRelevancyImpl()
        return self._modern_metric

    def _is_modern_llm(self) -> bool:
        """Detect if current LLM is modern (instructor-based)."""
        return (
            self.llm is not None
            and hasattr(self.llm, "instructor_llm")  # It's our compatibility wrapper
        )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """Same interface - routes to appropriate implementation."""

        if self._is_modern_llm():
            # Route to modern SimpleLLMMetric-based implementation
            return await self._modern_single_turn_ascore(sample, callbacks)
        else:
            # Route to legacy implementation
            return await self._legacy_single_turn_ascore(sample, callbacks)

    async def _legacy_single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """Original implementation - question generation + embeddings."""
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _modern_single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """Modern implementation - direct scoring via simple prompts."""
        # Validate required fields
        if sample.user_input is None:
            logger.warning("user_input is None, cannot score answer relevancy")
            return 0.0

        if sample.response is None:
            logger.warning("response is None, cannot score answer relevancy")
            return 0.0

        modern_metric = self._get_modern_metric()

        # Use modern metric with simple interface
        result = await modern_metric.ascore(
            question=sample.user_input, answer=sample.response, llm=self.llm
        )

        # Extract score from simple dictionary result
        try:
            score = float(result["score"])
            is_noncommittal = bool(result.get("is_noncommittal", False))

            # Apply noncommittal penalty
            return score * (0.0 if is_noncommittal else 1.0)

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Error extracting score from modern metric result: {e}")
            logger.warning(f"Result: {result}")

            # Final fallback
            if isinstance(result, dict) and "score" in result:
                try:
                    return float(result["score"])
                except (TypeError, ValueError):
                    pass

            return 0.0

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """Legacy scoring method - kept for backward compatibility."""
        return await super()._ascore(row, callbacks)


answer_relevancy = AnswerRelevancy()
