"""Answer Relevancy metric using SimplePydanticPrompt for easy modification and translation."""

import typing as t

import numpy as np
from pydantic import BaseModel

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.prompt.simple_mixin import SimplePromptMixin
from ragas.prompt.simple_pydantic_prompt import SimplePydanticPrompt

if t.TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM


# Input/Output models for the prompt
class AnswerRelevanceInput(BaseModel):
    """Input model for answer relevance evaluation."""

    response: str


class AnswerRelevanceOutput(BaseModel):
    """Output model for answer relevance evaluation."""

    question: str
    noncommittal: int


# The prompt definition using SimplePydanticPrompt
class AnswerRelevancePrompt(
    SimplePydanticPrompt[AnswerRelevanceInput, AnswerRelevanceOutput]
):
    """
    Prompt for generating questions from responses and detecting noncommittal answers.

    This prompt can be easily modified and translated using the SimplePromptMixin methods.
    """

    instruction = """Generate a question for the given answer and identify if the answer is noncommittal. 

Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. 
A noncommittal answer is one that is evasive, vague, or ambiguous. 
For example, "I don't know" or "I'm not sure" are noncommittal answers."""

    input_model = AnswerRelevanceInput
    output_model = AnswerRelevanceOutput
    name = "answer_relevance_prompt"

    examples = [
        (
            AnswerRelevanceInput(response="Albert Einstein was born in Germany."),
            AnswerRelevanceOutput(
                question="Where was Albert Einstein born?", noncommittal=0
            ),
        ),
        (
            AnswerRelevanceInput(
                response="I don't know about the groundbreaking feature of the smartphone invented in 2023 as I am unaware of information beyond 2022."
            ),
            AnswerRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]


class AnswerRelevancy(BaseMetric, SimplePromptMixin):
    """
    Evaluate answer relevancy by generating questions from the response and comparing to original question.

    This implementation uses SimplePydanticPrompt which supports:
    - Easy modification of prompts via get_prompts()/set_prompts()
    - Translation to different languages via adapt_prompts()
    - Clean prompt structure without bloat

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
        >>> # Modify the prompt instruction
        >>> metric.modify_prompt("answer_relevance_prompt",
        ...     instruction="Generate questions and detect evasive answers with extra care for technical topics.")
        >>>
        >>> # Translate prompts to Spanish
        >>> adapted_prompts = await metric.adapt_prompts("spanish", llm)
        >>> metric.set_adapted_prompts(adapted_prompts)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France."
        ... )
        >>> print(f"Score: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for question generation
        embeddings: Modern embeddings model with embed_text() and embed_texts() methods
        name: The metric name
        strictness: Number of questions to generate per answer (3-5 recommended)
        answer_relevance_prompt: The prompt used for evaluation (modifiable)
    """

    # Type hints for linter
    llm: "InstructorBaseRagasLLM"
    embeddings: "BaseRagasEmbedding"

    # The prompt attribute - this will be discovered by SimplePromptMixin
    answer_relevance_prompt: AnswerRelevancePrompt

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

        # Initialize the prompt
        self.answer_relevance_prompt = AnswerRelevancePrompt()

        # Call super() for validation
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, response: str) -> MetricResult:
        """
        Calculate answer relevancy score asynchronously.

        Args:
            user_input: The original question
            response: The response to evaluate

        Returns:
            MetricResult with relevancy score (0.0-1.0)
        """
        input_data = AnswerRelevanceInput(response=response)

        generated_questions = []
        noncommittal_flags = []

        # Generate multiple questions using the current prompt
        for _ in range(self.strictness):
            prompt_text = self.answer_relevance_prompt.to_string(input_data)
            result = await self.llm.agenerate(prompt_text, AnswerRelevanceOutput)

            if result.question:
                generated_questions.append(result.question)
                noncommittal_flags.append(result.noncommittal)

        if not generated_questions:
            return MetricResult(value=0.0)

        # Check if all responses were noncommittal
        all_noncommittal = np.all(noncommittal_flags)

        # Calculate similarity between original question and generated questions
        question_vec = np.asarray(self.embeddings.embed_text(user_input)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_texts(generated_questions)
        ).reshape(len(generated_questions), -1)

        # Calculate cosine similarity
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        cosine_sim = (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

        # Average similarity, penalized if all answers were noncommittal
        score = cosine_sim.mean() * int(not all_noncommittal)

        return MetricResult(value=float(score))
