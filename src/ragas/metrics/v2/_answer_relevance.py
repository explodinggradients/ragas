"""Answer Relevance metric implementation using modern instructor LLM."""

import typing as t

import numpy as np
from pydantic import BaseModel

from ragas.metrics.result import MetricResult
from ragas.prompt.metrics.answer_relevance import answer_relevance_prompt

if t.TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM


class AnswerRelevanceOutput(BaseModel):
    """Structured output for answer relevance question generation."""

    question: str
    noncommittal: int


async def answer_relevancy(
    user_input: str,
    response: str,
    llm: "InstructorBaseRagasLLM",
    embeddings: "BaseRagasEmbedding",
    strictness: int = 3,
) -> MetricResult:
    """
    Calculate answer relevancy by generating questions from the response and comparing to original question.

    Args:
        user_input: The original question
        response: The response to evaluate
        llm: instructor-based LLM for question generation.
             Use: instructor_llm_factory("openai", model="gpt-4o-mini", client=openai_client)
        embeddings: Modern embeddings model (BaseRagasEmbedding) with embed_text() and embed_texts() methods.
                   Use: embedding_factory("openai", model="text-embedding-ada-002", client=openai_client, interface="modern")
        strictness: Number of questions to generate (default: 3)

    Returns:
        MetricResult with relevancy score (0.0-1.0)
    """
    # Reject legacy wrappers
    # TODO: Should we support more llms / have a common new factory?
    if type(llm).__name__ != "InstructorLLM":
        raise ValueError(
            f"V2 metrics only support modern InstructorLLM. Found: {type(llm).__name__}. "
            f"Use: instructor_llm_factory('openai', model='gpt-4o-mini', client=openai_client)"
        )

    # TODO: Should we check for more legacy wrappers?
    if type(embeddings).__name__ == "LangchainEmbeddingsWrapper":
        raise ValueError(
            "V2 metrics only support modern embeddings. Legacy LangchainEmbeddingsWrapper is not supported. "
            "Use: embedding_factory('openai', model='text-embedding-ada-002', client=openai_client, interface='modern')"
        )

    prompt = answer_relevance_prompt(response)

    generated_questions = []
    noncommittal_flags = []

    for _ in range(strictness):
        result = await llm.agenerate(prompt, AnswerRelevanceOutput)

        if result.question:
            generated_questions.append(result.question)
            noncommittal_flags.append(result.noncommittal)

    if not generated_questions:
        return MetricResult(value=0.0)

    all_noncommittal = np.all(noncommittal_flags)

    question_vec = np.asarray(embeddings.embed_text(user_input)).reshape(1, -1)
    gen_question_vec = np.asarray(embeddings.embed_texts(generated_questions)).reshape(
        len(generated_questions), -1
    )

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
