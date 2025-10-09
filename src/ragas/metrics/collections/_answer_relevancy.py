"""Answer Relevancy metric v2 - Class-based implementation with modern components."""

import typing as t

import numpy as np
from pydantic import BaseModel

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.prompt.metrics.answer_relevance import answer_relevancy_prompt

if t.TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM


class AnswerRelevanceOutput(BaseModel):
    """Structured output for answer relevance question generation."""

    question: str
    noncommittal: int


class AnswerRelevancy(BaseMetric):
    """
    Evaluate answer relevancy by generating questions from the response and comparing to original question.

    This implementation uses modern instructor LLMs with structured output and modern embeddings.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import instructor_llm_factory
        >>> from ragas.embeddings.base import embedding_factory
        >>> from ragas.metrics.collections import AnswerRelevancy
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = instructor_llm_factory("openai", client=client, model="gpt-4o-mini")
        >>> embeddings = embedding_factory("openai", model="text-embedding-ada-002", client=client, interface="modern")
        >>>
        >>> # Create metric instance
        >>> metric = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France."
        ... )
        >>> print(f"Score: {result.value}")
        >>>
        >>> # Batch evaluation
        >>> results = await metric.abatch_score([
        ...     {"user_input": "Q1", "response": "A1"},
        ...     {"user_input": "Q2", "response": "A2"},
        ... ])

    Attributes:
        llm: Modern instructor-based LLM for question generation
        embeddings: Modern embeddings model with embed_text() and embed_texts() methods
        name: The metric name
        strictness: Number of questions to generate per answer (3-5 recommended)
        allowed_values: Score range (0.0 to 1.0)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"
    embeddings: "BaseRagasEmbedding"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        embeddings: "BaseRagasEmbedding",
        name: str = "answer_relevancy",
        strictness: int = 3,
        **kwargs,
    ):
        """Initialize AnswerRelevancy metric with required components."""
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.embeddings = embeddings
        self.strictness = strictness

        # Call super() for validation (without passing llm/embeddings in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, response: str) -> MetricResult:
        """
        Calculate answer relevancy score asynchronously.

        Components are guaranteed to be validated and non-None by the base class.

        Args:
            user_input: The original question
            response: The response to evaluate

        Returns:
            MetricResult with relevancy score (0.0-1.0)
        """
        prompt = answer_relevancy_prompt(response)

        generated_questions = []
        noncommittal_flags = []

        for _ in range(self.strictness):
            result = await self.llm.agenerate(prompt, AnswerRelevanceOutput)

            if result.question:
                generated_questions.append(result.question)
                noncommittal_flags.append(result.noncommittal)

        if not generated_questions:
            return MetricResult(value=0.0)

        all_noncommittal = np.all(noncommittal_flags)

        question_vec = np.asarray(self.embeddings.embed_text(user_input)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_texts(generated_questions)
        ).reshape(len(generated_questions), -1)

        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        cosine_sim = (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

        score = cosine_sim.mean() * int(not all_noncommittal)

        return MetricResult(value=float(score))
