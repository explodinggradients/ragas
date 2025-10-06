"""Answer Relevancy metric v2 with improved validation and bug fixes."""

import typing as t

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ragas.metrics.result import MetricResult
from ragas.metrics.v2.base import V2BaseMetric

if t.TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM


class AnswerRelevanceOutput(BaseModel):
    """Structured output for answer relevance question generation."""

    question: str = Field(..., description="Generated question from the answer")
    noncommittal: int = Field(
        ..., description="1 if answer is noncommittal, 0 if committal"
    )


class AnswerRelevancy(V2BaseMetric):
    """
    Evaluates answer relevancy by generating questions from the response.

    This metric generates questions from the given answer and compares them with
    the original question using embedding similarity.

    Attributes:
        name: The metric name
        llm: Instructor LLM for question generation
        embeddings: Embeddings model for similarity calculation
        strictness: Number of questions to generate (1-10, recommended 3-5)
        allowed_values: Score range (0.0 to 1.0)

    Example:
        >>> from ragas.metrics.v2 import AnswerRelevancy
        >>> from ragas.llms import instructor_llm_factory
        >>> from ragas.embeddings import embedding_factory
        >>>
        >>> llm = instructor_llm_factory("openai", model="gpt-4o-mini")
        >>> embeddings = embedding_factory("openai", model="text-embedding-ada-002", interface="modern")
        >>>
        >>> metric = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3)
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France."
        ... )
    """

    name: str = "answer_relevancy_v2"
    llm: "InstructorBaseRagasLLM" = Field(
        ..., description="Instructor LLM for question generation"
    )
    embeddings: "BaseRagasEmbedding" = Field(
        ..., description="Embeddings model for similarity calculation"
    )
    strictness: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of questions to generate (1-10, recommended 3-5)",
    )

    @field_validator("llm")
    @classmethod
    def validate_llm(cls, v: t.Any) -> "InstructorBaseRagasLLM":
        """Validate LLM compatibility."""
        if not hasattr(v, "agenerate"):
            raise ValueError(
                "LLM must support structured output generation. "
                "Use: instructor_llm_factory('openai', model='gpt-4o-mini')"
            )

        try:
            from ragas.llms.base import InstructorBaseRagasLLM

            if not isinstance(v, InstructorBaseRagasLLM):
                raise ValueError(
                    f"Incompatible LLM type: {type(v).__name__}. "
                    "Use: instructor_llm_factory('openai', model='gpt-4o-mini')"
                )
        except ImportError:
            pass

        return v

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v: t.Any) -> "BaseRagasEmbedding":
        """Validate embeddings compatibility."""
        if not hasattr(v, "embed_text") or not hasattr(v, "embed_texts"):
            raise ValueError(
                "Embeddings must support text embedding operations. "
                "Use: embedding_factory('openai', model='text-embedding-ada-002', interface='modern')"
            )

        return v

    async def _ascore_impl(self, user_input: str, response: str) -> MetricResult:
        """
        Calculate answer relevancy score asynchronously.

        Args:
            user_input: The original question
            response: The response to evaluate

        Returns:
            MetricResult with relevancy score (0.0-1.0)
        """
        prompt = self._create_prompt(response)

        generated_questions = []
        noncommittal_flags = []

        # Generate multiple questions for robustness
        for _ in range(self.strictness):
            try:
                result = await self.llm.agenerate(prompt, AnswerRelevanceOutput)

                if result and result.question:
                    generated_questions.append(result.question)
                    noncommittal_flags.append(result.noncommittal)
            except Exception:
                continue

        if not generated_questions:
            return MetricResult(
                value=0.0, reason="Failed to generate any questions from the response"
            )

        all_noncommittal = all(noncommittal_flags)

        # Compute embeddings
        question_vec = np.asarray(self.embeddings.embed_text(user_input)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_texts(generated_questions)
        ).reshape(len(generated_questions), -1)

        # Calculate cosine similarity
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )

        epsilon = 1e-10
        cosine_sim = np.dot(gen_question_vec, question_vec.T).reshape(
            -1,
        ) / (norm + epsilon)

        score = float(cosine_sim.mean()) * int(not all_noncommittal)

        return MetricResult(value=score)

    def _create_prompt(self, response: str) -> str:
        """
        Generate the prompt for answer relevance evaluation.

        Args:
            response: The response text to evaluate

        Returns:
            Formatted prompt string for the LLM
        """
        import json

        escaped_response = json.dumps(response)

        return f"""Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers

Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"properties": {{"question": {{"title": "Question", "type": "string"}}, "noncommittal": {{"title": "Noncommittal", "type": "integer"}}}}, "required": ["question", "noncommittal"], "title": "ResponseRelevanceOutput", "type": "object"}}

--------EXAMPLES-----------
Example 1
Input: {{
    "response": "Albert Einstein was born in Germany."
}}
Output: {{
    "question": "Where was Albert Einstein born?",
    "noncommittal": 0
}}

Example 2
Input: {{
    "response": "I don't know about the groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022."
}}
Output: {{
    "question": "What was the groundbreaking feature of the smartphone invented in 2023?",
    "noncommittal": 1
}}
-----------------------------

Now perform the same with the following input
input: {{
    "response": {escaped_response}
}}
Output: """
