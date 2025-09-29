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
from ragas.prompt.metric_prompts import NLI_STATEMENT_PROMPT, STATEMENT_GENERATOR_PROMPT

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
# CENTRALIZED PROMPTS (Imported from ragas.prompt.metric_prompts)
# ============================================================================

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
        result = self.llm.generate(  # type: ignore
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
        result = self.llm.generate(prompt, response_model=NLIStatementOutput)  # type: ignore

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
