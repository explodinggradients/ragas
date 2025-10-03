"""Answer Relevancy metric using modern LLM and embedding interfaces."""

import typing as t
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from ragas.async_utils import run
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.result import MetricResult
from ragas.metrics.validators import NumericValidator


# Pydantic models for structured LLM outputs
class QuestionGenerationOutput(BaseModel):
    """Output model for question generation from response."""

    question: str
    noncommittal: int  # 1 if answer is noncommittal/evasive, 0 otherwise


# Default instruction with few-shot examples
DEFAULT_INSTRUCTION = """Generate a question for the given answer and identify if the answer is noncommittal.

Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal.
A noncommittal answer is one that is evasive, vague, or ambiguous.
For example, "I don't know" or "I'm not sure" are noncommittal answers.

Examples:

Answer: Albert Einstein was born in Germany.
Output: {{"question": "Where was Albert Einstein born?", "noncommittal": 0}}

Answer: I don't know about the groundbreaking feature of the smartphone invented in 2023 as I am unaware of information beyond 2022.
Output: {{"question": "What was the groundbreaking feature of the smartphone invented in 2023?", "noncommittal": 1}}

Now generate for the following answer:"""


@dataclass
class AnswerRelevancy(SimpleBaseMetric, NumericValidator):
    """
    Evaluates relevancy of response to user input using LLM and embeddings.

    This metric measures how relevant the response is to the user's question by:
    1. Generating N questions from the response using an LLM
    2. Calculating cosine similarity between generated questions and original question
    3. Penalizing noncommittal/evasive answers

    The score ranges from 0.0 (not relevant) to 1.0 (highly relevant).

    Parameters
    ----------
    llm : InstructorBaseRagasLLM
        LLM for generating questions from responses (required)
    embeddings : BaseRagasEmbedding
        Embeddings model for calculating similarity (required)
    strictness : int, default=3
        Number of questions to generate per response (3-5 recommended)
    instruction : str, optional
        Custom instruction for question generation (has good defaults with examples)
    name : str, default="answer_relevancy"
        Name of the metric
    allowed_values : tuple, default=(0.0, 1.0)
        Valid range for metric scores

    Examples
    --------
    >>> from ragas.metrics.v2 import AnswerRelevancy
    >>> from ragas.llms import instructor_llm_factory
    >>> from ragas.embeddings import embedding_factory
    >>> import instructor
    >>> from openai import OpenAI
    >>>
    >>> # Setup dependencies
    >>> client = instructor.from_openai(OpenAI())
    >>> llm = instructor_llm_factory("openai", client=client, model="gpt-4o-mini")
    >>> embeddings = embedding_factory("openai")
    >>>
    >>> # Create metric instance
    >>> metric = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3)
    >>>
    >>> # Single evaluation
    >>> result = await metric.ascore(
    ...     user_input="What is the capital of France?",
    ...     response="Paris is the capital of France."
    ... )
    >>> print(f"Score: {result.value}, Reason: {result.reason}")
    >>>
    >>> # Batch evaluation
    >>> results = await metric.abatch_score([
    ...     {"user_input": "Q1", "response": "A1"},
    ...     {"user_input": "Q2", "response": "A2"},
    ... ])
    """

    # Fields with defaults (must come first due to dataclass ordering)
    name: str = "answer_relevancy"
    allowed_values: t.Tuple[float, float] = (0.0, 1.0)
    strictness: int = 3
    instruction: str = DEFAULT_INSTRUCTION

    # Required fields - use sentinel to make them required
    llm: t.Optional[InstructorBaseRagasLLM] = None
    embeddings: t.Optional[BaseRagasEmbedding] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dependencies
        if self.llm is None:
            raise ValueError("llm is required for AnswerRelevancy metric")
        if self.embeddings is None:
            raise ValueError("embeddings is required for AnswerRelevancy metric")

        # Validate strictness
        if self.strictness < 1:
            raise ValueError(f"strictness must be >= 1, got {self.strictness}")

    def _build_prompt(self, response: str) -> str:
        """Build prompt for question generation."""
        return f"{self.instruction}\n\nAnswer: {response}"

    async def _generate_questions(
        self, response: str
    ) -> t.List[QuestionGenerationOutput]:
        """Generate multiple questions from response using LLM."""
        prompt = self._build_prompt(response)

        # Generate multiple questions concurrently
        # (InstructorLLM doesn't have native n-completion, so call multiple times)
        import asyncio

        tasks = [
            self.llm.agenerate(prompt, QuestionGenerationOutput)
            for _ in range(self.strictness)
        ]

        results = await asyncio.gather(*tasks)
        return results

    async def _calculate_similarity_async(
        self, question: str, generated_questions: t.List[str]
    ) -> np.ndarray:
        """Calculate cosine similarity between questions using embeddings (async)."""
        # Embed all questions concurrently
        question_embedding = await self.embeddings.aembed_text(question)
        gen_embeddings = await self.embeddings.aembed_texts(generated_questions)

        # Convert to numpy arrays
        question_vec = np.asarray(question_embedding).reshape(1, -1)
        gen_question_vec = np.asarray(gen_embeddings).reshape(
            len(generated_questions), -1
        )

        # Calculate cosine similarity
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )

        # Avoid division by zero
        norm = np.where(norm == 0, 1e-10, norm)

        cosine_sim = (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

        return cosine_sim

    async def ascore(self, user_input: str, response: str, **kwargs) -> MetricResult:
        """
        Calculate answer relevancy score asynchronously.

        Parameters
        ----------
        user_input : str
            The original question/query from the user
        response : str
            The response/answer to evaluate

        Returns
        -------
        MetricResult
            Score between 0.0 and 1.0, where 1.0 is most relevant.
            Includes detailed reasoning about the evaluation.
        """
        try:
            # Generate questions from response
            generated = await self._generate_questions(response)

            # Extract questions and check for noncommittal answers
            gen_questions = [g.question for g in generated]
            all_noncommittal = all(g.noncommittal for g in generated)

            # Handle edge case: all generated questions are empty
            if all(q.strip() == "" for q in gen_questions):
                return MetricResult(
                    value=np.nan,
                    reason="All generated questions were empty. Response may be invalid.",
                )

            # Calculate similarity between original and generated questions
            cosine_sim = await self._calculate_similarity_async(
                user_input, gen_questions
            )
            mean_sim = float(cosine_sim.mean())

            # Penalize noncommittal answers (multiply by 0 if all are noncommittal)
            score = mean_sim * int(not all_noncommittal)

            # Build detailed reasoning
            reason = (
                f"Generated {len(gen_questions)} questions. "
                f"Mean cosine similarity: {mean_sim:.3f}. "
                f"All noncommittal: {all_noncommittal}"
            )

            return MetricResult(value=score, reason=reason)

        except Exception as e:
            # Return error as MetricResult for better error tracking
            return MetricResult(
                value=None, reason=f"Error calculating answer relevancy: {str(e)}"
            )

    def score(self, user_input: str, response: str, **kwargs) -> MetricResult:
        """
        Calculate answer relevancy score synchronously.

        This is a convenience wrapper around ascore() that handles event loops
        automatically (including Jupyter notebooks).

        Parameters
        ----------
        user_input : str
            The original question/query from the user
        response : str
            The response/answer to evaluate

        Returns
        -------
        MetricResult
            Score between 0.0 and 1.0, where 1.0 is most relevant
        """
        return run(self.ascore(user_input, response, **kwargs))
