from __future__ import annotations

import asyncio
import json
import logging
import typing as t
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.metrics.utils import fbeta_score
from ragas.prompt.metric_prompts import NLI_STATEMENT_PROMPT

if t.TYPE_CHECKING:
    from ragas.dataset_schema import SingleTurnSample

T = t.TypeVar("T")
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS (No LangChain dependencies)
# ============================================================================


class ClaimDecompositionInput(BaseModel):
    response: str = Field(..., title="Response")


class ClaimDecompositionOutput(BaseModel):
    claims: t.List[str] = Field(..., title="Decomposed Claims")


# ============================================================================
# DECOMPOSITION TYPES AND EXAMPLES
# ============================================================================


class DecompositionType(Enum):
    LOW_ATOMICITY_LOW_COVERAGE = "low_atomicity_low_coverage"
    LOW_ATOMICITY_HIGH_COVERAGE = "low_atomicity_high_coverage"
    HIGH_ATOMICITY_LOW_COVERAGE = "high_atomicity_low_coverage"
    HIGH_ATOMICITY_HIGH_COVERAGE = "high_atomicity_high_coverage"


# Example input data
example1_input = ClaimDecompositionInput(
    response="Charles Babbage was a French mathematician, philosopher, and food critic."
)

# Define the examples using the Pydantic structure
claim_decomposition_examples = {
    DecompositionType.LOW_ATOMICITY_LOW_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                claims=["Charles Babbage was a mathematician and philosopher."]
            ),
        )
    ],
    DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                claims=[
                    "Charles Babbage was a French mathematician, philosopher, and food critic."
                ]
            ),
        )
    ],
    DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                claims=[
                    "Charles Babbage was a mathematician.",
                    "Charles Babbage was a philosopher.",
                ]
            ),
        )
    ],
    DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                claims=[
                    "Charles Babbage was a mathematician.",
                    "Charles Babbage was a philosopher.",
                    "Charles Babbage was a food critic.",
                    "Charles Babbage was French.",
                ]
            ),
        )
    ],
}

# Example input data with two sentences
example2_input = ClaimDecompositionInput(
    response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics."
)

# Adding examples to the dictionary with different decomposition types
claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            claims=[
                "Albert Einstein was a German physicist.",
                "Albert Einstein developed relativity and contributed to quantum mechanics.",
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            claims=[
                "Albert Einstein was a German theoretical physicist.",
                "Albert Einstein developed the theory of relativity and also contributed to the development of quantum mechanics.",
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            claims=[
                "Albert Einstein was a German theoretical physicist.",
                "Albert Einstein developed the theory of relativity.",
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            claims=[
                "Albert Einstein was a German theoretical physicist.",
                "Albert Einstein developed the theory of relativity.",
                "Albert Einstein contributed to the development of quantum mechanics.",
            ]
        ),
    )
)


# ============================================================================
# DIRECT PROMPT TEMPLATES (No PydanticPrompt dependencies)
# ============================================================================


def _generate_claim_decomposition_prompt(
    atomicity: str, coverage: str, response: str
) -> str:
    """Generate claim decomposition prompt based on atomicity and coverage levels."""

    # Get examples for the specified atomicity and coverage
    decomposition_type = DecompositionType(f"{atomicity}_atomicity_{coverage}_coverage")
    examples = claim_decomposition_examples.get(decomposition_type, [])

    # Build examples section
    examples_text = ""
    if examples:
        examples_text = "\n--------EXAMPLES-----------\n"
        for i, (input_example, output_example) in enumerate(examples, 1):
            examples_text += f"Example {i}\n"
            examples_text += f'Input: {{"response": "{input_example.response}"}}\n'
            examples_text += (
                f'Output: {{"claims": {json.dumps(output_example.claims)}}}\n\n'
            )
        examples_text += "-----------------------------\n"

    # Use the centralized prompt template with dynamic examples
    return f"""Decompose and break down each of the input sentences into one or more standalone statements. Each statement should be a standalone claim that can be independently verified.
Follow the level of atomicity and coverage as shown in the examples.
{examples_text}
Now perform the same with the following input
input: {{"response": "{response}"}}
Output: """


# NLI prompt imported from centralized location at top of file

# ============================================================================
# MIGRATED FACTUAL CORRECTNESS METRIC (No LangChain dependencies)
# ============================================================================


@dataclass
class FactualCorrectness(MetricWithLLM, SingleTurnMetric):
    """
    FactualCorrectness metric without LangChain dependencies.

    Evaluates the factual correctness of responses generated by a language model.
    It uses claim decomposition and natural language inference (NLI) to verify
    the claims made in the responses against reference texts.

    Key changes from the original implementation:
    - Removed LangChain callback dependencies
    - Uses direct string-based prompts instead of PydanticPrompt classes
    - Simplified LLM interface calls
    - Maintains the same scoring logic and behavior
    - Improved JSON parsing with better error handling

    Attributes:
        name (str): The name of the metric, default is "factual_correctness".
        _required_columns (Dict[MetricType, Set[str]]): A dictionary specifying the required columns
            for each metric type. Default is {"SINGLE_TURN": {"response", "reference"}}.
        mode (Literal["precision", "recall", "f1"]): The mode of evaluation, can be "precision",
            "recall", or "f1". Default is "f1".
        beta (float): The beta value used for the F1 score calculation. A beta > 1 gives more weight
            to recall, while beta < 1 favors precision. Default is 1.0.
        atomicity (Literal["low", "high"]): The level of atomicity for claim decomposition. Default is "low".
        coverage (Literal["low", "high"]): The level of coverage for claim decomposition. Default is "low".
    """

    name: str = "factual_correctness"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS
    mode: t.Literal["precision", "recall", "f1"] = "f1"
    beta: float = 1.0
    atomicity: t.Literal["low", "high"] = "low"
    coverage: t.Literal["low", "high"] = "low"
    language: str = "english"

    def __post_init__(self):
        if type(self.beta) is not float:
            raise ValueError(
                "Beta must be a float. A beta > 1 gives more weight to recall, while beta < 1 favors precision."
            )

    async def decompose_claims(self, response: str) -> t.List[str]:
        """Decompose response into claims using direct LLM call."""
        assert self.llm is not None, "LLM must be set"

        prompt = _generate_claim_decomposition_prompt(
            self.atomicity, self.coverage, response
        )

        # Use Instructor LLM interface for direct API calls without LangChain
        result = await self.llm.agenerate(  # type: ignore
            prompt, response_model=ClaimDecompositionOutput
        )

        # Instructor returns structured objects directly - no JSON parsing needed!
        return result.claims

    async def verify_claims(
        self, premise: str, hypothesis_list: t.List[str]
    ) -> np.ndarray:
        """Verify claims using NLI with direct LLM call."""
        assert self.llm is not None, "LLM must be set"

        if not hypothesis_list:
            return np.array([], dtype=bool)

        statements_json = json.dumps(hypothesis_list)
        prompt = NLI_STATEMENT_PROMPT.format(
            context=premise, statements_json=statements_json
        )

        # Use Instructor LLM interface for direct API calls without LangChain
        from ragas.metrics._faithfulness import NLIStatementOutput

        result = await self.llm.agenerate(prompt, response_model=NLIStatementOutput)  # type: ignore

        # Instructor returns structured objects directly - no JSON parsing needed!
        verdicts = [bool(stmt.verdict) for stmt in result.statements]
        return np.array(verdicts, dtype=bool)

    @staticmethod
    async def _get_passthrough_value(value: T) -> T:
        """Utility method for async passthrough."""
        return value

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        """Score a single turn sample (callbacks parameter kept for compatibility but ignored)."""
        reference = sample.reference
        response = sample.response
        assert self.llm is not None, "LLM must be set"
        assert reference is not None, "Reference is not set"
        assert response is not None, "Response is not set"

        reference_response_task = self.decompose_and_verify_claims(reference, response)

        if self.mode != "precision":
            response_reference_task = self.decompose_and_verify_claims(
                response, reference
            )
        else:
            response_reference_task = self._get_passthrough_value(
                value=np.array([], dtype=bool)
            )

        reference_response, response_reference = await asyncio.gather(
            reference_response_task, response_reference_task
        )

        tp = sum(reference_response)
        fp = sum(~reference_response)
        if self.mode != "precision":
            fn = sum(~response_reference)
        else:
            fn = 0

        if self.mode == "precision":
            score = tp / (tp + fp + 1e-8)
        elif self.mode == "recall":
            score = tp / (tp + fn + 1e-8)
        else:
            score = fbeta_score(tp, fp, fn, self.beta)

        return np.round(score, 2)

    async def decompose_and_verify_claims(
        self, reference: str, response: str
    ) -> np.ndarray:
        """Decompose claims and verify them against reference."""
        claims = await self.decompose_claims(response)
        return await self.verify_claims(premise=reference, hypothesis_list=claims)

    async def _ascore(self, row: t.Dict, callbacks=None) -> float:
        """Calculate factual correctness score."""
        from ragas.dataset_schema import SingleTurnSample

        return await self._single_turn_ascore(SingleTurnSample(**row))


# Create default instance
factual_correctness = FactualCorrectness()
