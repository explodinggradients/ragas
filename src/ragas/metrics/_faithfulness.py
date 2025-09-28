from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS (No LangChain dependencies)
# ============================================================================


class StatementGeneratorInput(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")


class StatementGeneratorOutput(BaseModel):
    statements: t.List[str] = Field(description="The generated statements")


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class NLIStatementOutput(BaseModel):
    statements: t.List[StatementFaithfulnessAnswer]


class NLIStatementInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statements: t.List[str] = Field(..., description="The statements to judge")


# ============================================================================
# MINIMAL COMPATIBILITY STUBS (No LangChain dependencies)
# ============================================================================
# These are minimal stub classes for backward compatibility with other metrics.
# They don't actually work - metrics using them should be migrated to use
# direct prompt templates or their own LangChain-free implementations.


class _DeprecatedPromptStub:
    """Minimal stub class for backward compatibility. Does not work - migrate to direct prompts."""

    def __init__(self):
        raise NotImplementedError(
            "PydanticPrompt classes have been removed to eliminate LangChain dependencies. "
            "Please migrate to use direct prompt templates or implement your own LangChain-free version."
        )


# Create aliases for backward compatibility
NLIStatementPrompt = _DeprecatedPromptStub
StatementGeneratorPrompt = _DeprecatedPromptStub


# ============================================================================
# DIRECT PROMPT TEMPLATES (No PydanticPrompt dependencies)
# ============================================================================

STATEMENT_GENERATOR_PROMPT = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.

--------EXAMPLES-----------
Example 1
Input: {{"question": "Who was Albert Einstein and what is he best known for?", "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."}}
Output: {{"statements": ["Albert Einstein was a German-born theoretical physicist.", "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.", "Albert Einstein was best known for developing the theory of relativity.", "Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]}}
-----------------------------

Now perform the same with the following input
input: {{"question": "{question}", "answer": "{answer}"}}
Output: """

NLI_STATEMENT_PROMPT = """Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.

--------EXAMPLES-----------
Example 1
Input: {{"context": "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.", "statements": ["John is majoring in Biology.", "John is taking a course on Artificial Intelligence.", "John is a dedicated student.", "John has a part-time job."]}}
Output: {{"statements": [{{"statement": "John is majoring in Biology.", "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.", "verdict": 0}}, {{"statement": "John is taking a course on Artificial Intelligence.", "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.", "verdict": 0}}, {{"statement": "John is a dedicated student.", "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.", "verdict": 1}}, {{"statement": "John has a part-time job.", "reason": "There is no information given in the context about John having a part-time job.", "verdict": 0}}]}}

Example 2
Input: {{"context": "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.", "statements": ["Albert Einstein was a genius."]}}
Output: {{"statements": [{{"statement": "Albert Einstein was a genius.", "reason": "The context and statement are unrelated", "verdict": 0}}]}}
-----------------------------

Now perform the same with the following input
input: {{"context": "{context}", "statements": {statements_json}}}
Output: """


# ============================================================================
# MIGRATED FAITHFULNESS METRIC (No LangChain dependencies)
# ============================================================================


@dataclass
class Faithfulness(MetricWithLLM, SingleTurnMetric):
    """
    Faithfulness metric without LangChain dependencies.

    The Faithfulness metric measures how factually consistent a response is with the
    retrieved context. It ranges from 0 to 1, with higher scores indicating better consistency.

    Key changes from the original implementation:
    - Removed LangChain callback dependencies
    - Uses direct string-based prompts instead of PydanticPrompt classes
    - Simplified LLM interface calls
    - Maintains the same scoring logic and behavior
    - Improved JSON parsing with better error handling
    """

    name: str = "faithfulness"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts",
            }
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS
    max_retries: int = 1

    async def _create_statements(self, row: t.Dict) -> StatementGeneratorOutput:
        """Generate statements from response using direct LLM call."""
        assert self.llm is not None, "llm is not set"

        question = row["user_input"]
        answer = row["response"]

        prompt = STATEMENT_GENERATOR_PROMPT.format(question=question, answer=answer)

        # Use Instructor LLM interface for direct API calls without LangChain
        result = await self.llm.agenerate(
            prompt, response_model=StatementGeneratorOutput
        )

        # Instructor returns structured objects directly - no JSON parsing needed!
        return result

    async def _create_verdicts(
        self, row: t.Dict, statements: t.List[str]
    ) -> NLIStatementOutput:
        """Create verdicts for statements using direct LLM call."""
        assert self.llm is not None, "llm must be set to compute score"

        contexts_str = "\n".join(row["retrieved_contexts"])
        statements_json = json.dumps(statements)

        prompt = NLI_STATEMENT_PROMPT.format(
            context=contexts_str, statements_json=statements_json
        )

        # Use Instructor LLM interface for direct API calls without LangChain
        result = await self.llm.agenerate(prompt, response_model=NLIStatementOutput)

        # Instructor returns structured objects directly - no JSON parsing needed!
        return result

    def _compute_score(self, answers: NLIStatementOutput) -> float:
        """Compute faithfulness score from verdicts."""
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.statements
        )
        num_statements = len(answers.statements)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        """Score a single turn sample (callbacks parameter kept for compatibility but ignored)."""
        row = sample.to_dict()
        return await self._ascore(row)

    async def _ascore(self, row: t.Dict, callbacks=None) -> float:
        """
        Calculate faithfulness score.
        Returns the NLI score for each (question, context, answer) pair.
        """
        assert self.llm is not None, "LLM is not set"

        statements = await self._create_statements(row)
        if not statements.statements:
            return np.nan

        verdicts = await self._create_verdicts(row, statements.statements)
        return self._compute_score(verdicts)


@dataclass
class FaithfulnesswithHHEM(Faithfulness):
    """
    Faithfulness metric using Vectara's HHEM-2.1-Open model for NLI evaluation.

    This version uses a specialized hallucination detection model instead of LLM calls
    for the Natural Language Inference step, making it more efficient and cost-effective.
    """

    name: str = "faithfulness_with_hhem"
    device: str = "cpu"
    batch_size: int = 10

    def __post_init__(self):
        try:
            from transformers import AutoModelForSequenceClassification  # type: ignore
        except ImportError:
            raise ImportError(
                "Huggingface transformers must be installed to use this feature, try `pip install transformers`"
            )
        self.nli_classifier = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        )
        self.nli_classifier.to(self.device)
        super().__post_init__()

    def _create_pairs(
        self, row: t.Dict, statements: t.List[str]
    ) -> t.List[t.Tuple[str, str]]:
        """Create pairs of (premise, hypothesis) from the row."""
        premise = "\n".join(row["retrieved_contexts"])
        pairs = [(premise, statement) for statement in statements]
        return pairs

    def _create_batch(
        self, pairs: t.List[t.Tuple[str, str]]
    ) -> t.Generator[t.List[t.Tuple[str, str]], None, None]:
        """Create batches of pairs to avoid OOM."""
        length_of_pairs = len(pairs)
        for ndx in range(0, length_of_pairs, self.batch_size):
            yield pairs[ndx : min(ndx + self.batch_size, length_of_pairs)]

    async def _ascore(self, row: t.Dict, callbacks=None) -> float:
        """
        Calculate faithfulness score using HHEM model.
        Returns the NLI score for each (question, context, answer) pair.
        """
        assert self.llm is not None, "LLM is not set"

        statements = await self._create_statements(row)
        if not statements.statements:
            return np.nan

        scores = []
        pairs = self._create_pairs(row, statements.statements)
        for input_pairs in self._create_batch(pairs):  # to avoid OOM
            batch_scores = (
                self.nli_classifier.predict(input_pairs).cpu().detach().round()
            )
            # convert tensor to list of floats
            scores.extend(batch_scores.tolist())

        return sum(scores) / len(scores)


# Create default instances
faithfulness = Faithfulness()
