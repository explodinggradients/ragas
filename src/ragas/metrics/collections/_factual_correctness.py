"""Factual Correctness metric v2 - Modern implementation with multi-modal scoring."""

import typing as t
from typing import List

import numpy as np
from pydantic import BaseModel

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.metrics.utils import fbeta_score
from ragas.prompt.metrics.common import nli_statement_prompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class ClaimDecompositionOutput(BaseModel):
    """Structured output for claim decomposition."""

    claims: List[str]


class StatementFaithfulnessAnswer(BaseModel):
    """Individual statement with reason and verdict for NLI evaluation."""

    statement: str
    reason: str
    verdict: int


class NLIStatementOutput(BaseModel):
    """Structured output for NLI statement evaluation."""

    statements: List[StatementFaithfulnessAnswer]


def claim_decomposition_prompt(
    response: str, atomicity: str = "low", coverage: str = "low"
) -> str:
    """
    V1-identical claim decomposition prompt with configurable atomicity/coverage.

    Args:
        response: The response text to break down into claims
        atomicity: Level of atomicity ("low" or "high")
        coverage: Level of coverage ("low" or "high")

    Returns:
        V1-identical prompt string for the LLM
    """
    import json

    safe_response = json.dumps(response)

    # Select examples based on atomicity and coverage configuration
    if atomicity == "low" and coverage == "low":
        examples = [
            {
                "input": {
                    "response": "Charles Babbage was a French mathematician, philosopher, and food critic."
                },
                "output": {
                    "claims": ["Charles Babbage was a mathematician and philosopher."]
                },
            },
            {
                "input": {
                    "response": "Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics."
                },
                "output": {
                    "claims": [
                        "Albert Einstein was a German physicist.",
                        "Albert Einstein developed relativity and contributed to quantum mechanics.",
                    ]
                },
            },
        ]
    elif atomicity == "low" and coverage == "high":
        examples = [
            {
                "input": {
                    "response": "Charles Babbage was a French mathematician, philosopher, and food critic."
                },
                "output": {
                    "claims": [
                        "Charles Babbage was a French mathematician, philosopher, and food critic."
                    ]
                },
            },
            {
                "input": {
                    "response": "Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics."
                },
                "output": {
                    "claims": [
                        "Albert Einstein was a German theoretical physicist.",
                        "Albert Einstein developed the theory of relativity and also contributed to the development of quantum mechanics.",
                    ]
                },
            },
        ]
    elif atomicity == "high" and coverage == "low":
        examples = [
            {
                "input": {
                    "response": "Charles Babbage was a French mathematician, philosopher, and food critic."
                },
                "output": {
                    "claims": [
                        "Charles Babbage was a mathematician.",
                        "Charles Babbage was a philosopher.",
                    ]
                },
            },
            {
                "input": {
                    "response": "Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics."
                },
                "output": {
                    "claims": [
                        "Albert Einstein was a German theoretical physicist.",
                        "Albert Einstein developed the theory of relativity.",
                    ]
                },
            },
        ]
    else:  # high atomicity, high coverage
        examples = [
            {
                "input": {
                    "response": "Charles Babbage was a French mathematician, philosopher, and food critic."
                },
                "output": {
                    "claims": [
                        "Charles Babbage was a mathematician.",
                        "Charles Babbage was a philosopher.",
                        "Charles Babbage was a food critic.",
                        "Charles Babbage was French.",
                    ]
                },
            },
            {
                "input": {
                    "response": "Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics."
                },
                "output": {
                    "claims": [
                        "Albert Einstein was a German theoretical physicist.",
                        "Albert Einstein developed the theory of relativity.",
                        "Albert Einstein contributed to the development of quantum mechanics.",
                    ]
                },
            },
        ]

    # Build examples string
    examples_str = "\n".join(
        [
            f"""Example {i + 1}
Input: {json.dumps(ex["input"], indent=4)}
Output: {json.dumps(ex["output"], indent=4)}"""
            for i, ex in enumerate(examples)
        ]
    )

    return f"""Decompose and break down each of the input sentences into one or more standalone statements. Each statement should be a standalone claim that can be independently verified.
Follow the level of atomicity and coverage as shown in the examples.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"properties": {{"claims": {{"description": "Decomposed Claims", "items": {{"type": "string"}}, "title": "Claims", "type": "array"}}}}, "required": ["claims"], "title": "ClaimDecompositionOutput", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
{examples_str}
-----------------------------

Now perform the same with the following input
input: {{
    "response": {safe_response}
}}
Output: """


class FactualCorrectness(BaseMetric):
    """
    Modern v2 implementation of factual correctness evaluation.

    Evaluates the factual correctness of responses by comparing claims made in the response
    against a reference text. Uses claim decomposition and natural language inference (NLI)
    to verify claims in both directions.

    The metric supports three evaluation modes:
    - Precision: What fraction of response claims are supported by reference
    - Recall: What fraction of reference claims are covered by response
    - F1: Harmonic mean of precision and recall (with configurable beta)

    The metric also supports configurable claim decomposition:
    - Atomicity: "low" (fewer, broader claims) vs "high" (more, atomic claims)
    - Coverage: "low" (partial coverage) vs "high" (comprehensive coverage)

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import FactualCorrectness
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = FactualCorrectness(llm=llm, mode="f1", beta=1.0)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     response="Einstein was born in Germany in 1879.",
        ...     reference="Albert Einstein was born in Ulm, Germany on March 14, 1879."
        ... )
        >>> print(f"Factual Correctness: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for claim decomposition and NLI evaluation
        mode: Evaluation mode ("precision", "recall", or "f1")
        beta: Beta parameter for F1 score (>1 favors recall, <1 favors precision)
        atomicity: Claim decomposition atomicity ("low" or "high")
        coverage: Claim decomposition coverage ("low" or "high")
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        mode: t.Literal["precision", "recall", "f1"] = "f1",
        beta: float = 1.0,
        atomicity: t.Literal["low", "high"] = "low",
        coverage: t.Literal["low", "high"] = "low",
        name: str = "factual_correctness",
        **kwargs,
    ):
        """
        Initialize FactualCorrectness metric with required components.

        Args:
            llm: Modern instructor-based LLM for claim decomposition and NLI evaluation
            mode: Evaluation mode ("precision", "recall", or "f1")
            beta: Beta parameter for F1 score (>1 favors recall, <1 favors precision)
            atomicity: Claim decomposition atomicity ("low" or "high")
            coverage: Claim decomposition coverage ("low" or "high")
            name: The metric name
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.mode = mode
        self.beta = beta
        self.atomicity = atomicity
        self.coverage = coverage

        # Validate beta parameter
        if not isinstance(beta, (int, float)):
            raise ValueError(
                "Beta must be a float. A beta > 1 gives more weight to recall, while beta < 1 favors precision."
            )

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(self, response: str, reference: str) -> MetricResult:
        """
        Calculate factual correctness score.

        Args:
            response: The response to evaluate for factual correctness
            reference: The reference text to check claims against

        Returns:
            MetricResult with factual correctness score (0.0-1.0, higher is better)
        """
        # Input validation
        if not response:
            raise ValueError(
                "response is missing. Please add response to the test sample."
            )
        if not reference:
            raise ValueError(
                "reference is missing. Please add reference to the test sample."
            )

        # Step 1: Get claim verifications based on mode
        if self.mode != "precision":
            # For recall and f1: response claims → reference verification
            response_verified = await self._decompose_and_verify_claims(
                response, reference
            )
        else:
            response_verified = np.array([], dtype=bool)

        if self.mode != "recall":
            # For precision and f1: reference claims → response verification
            reference_verified = await self._decompose_and_verify_claims(
                reference, response
            )
        else:
            reference_verified = np.array([], dtype=bool)

        # Step 2: Compute TP, FP, FN
        if self.mode != "precision":
            tp = int(np.sum(response_verified))
            fn = int(np.sum(~response_verified))
        else:
            tp = int(np.sum(reference_verified))
            fn = 0

        if self.mode != "recall":
            fp = int(np.sum(~reference_verified))
        else:
            fp = 0

        # Step 3: Compute final score based on mode
        if self.mode == "precision":
            score = tp / (tp + fp + 1e-8)
        elif self.mode == "recall":
            score = tp / (tp + fn + 1e-8)
        else:  # f1
            score = fbeta_score(tp, fp, fn, self.beta)

        return MetricResult(value=float(np.round(score, 2)))

    async def _decompose_claims(self, response: str) -> List[str]:
        """Break response into claims using configurable decomposition."""
        prompt = claim_decomposition_prompt(
            response, atomicity=self.atomicity, coverage=self.coverage
        )
        result = await self.llm.agenerate(prompt, ClaimDecompositionOutput)
        return result.claims

    async def _verify_claims(
        self, claims: List[str], reference: str
    ) -> NLIStatementOutput:
        """Verify claims against reference using NLI."""
        prompt = nli_statement_prompt(reference, claims)
        result = await self.llm.agenerate(prompt, NLIStatementOutput)
        return result

    async def _decompose_and_verify_claims(
        self, text_to_decompose: str, reference_text: str
    ) -> np.ndarray:
        """Decompose text into claims and verify against reference."""
        claims = await self._decompose_claims(text_to_decompose)
        if not claims:
            return np.array([], dtype=bool)

        verdicts = await self._verify_claims(claims, reference_text)
        if not verdicts.statements:
            return np.array([], dtype=bool)

        return np.array([bool(stmt.verdict) for stmt in verdicts.statements])
