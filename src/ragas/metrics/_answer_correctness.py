from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness import StatementGeneratorOutput
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.metrics.utils import fbeta_score
from ragas.prompt.metric_prompts import (
    CORRECTNESS_CLASSIFIER_PROMPT,
    STATEMENT_GENERATOR_PROMPT,
)
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS (No LangChain dependencies)
# ============================================================================


class QuestionAnswerGroundTruth(BaseModel):
    question: str
    answer: list[str]
    ground_truth: list[str]


class StatementsWithReason(BaseModel):
    statement: str
    reason: str


class ClassificationWithReason(BaseModel):
    TP: list[StatementsWithReason]
    FP: list[StatementsWithReason]
    FN: list[StatementsWithReason]


# Prompts imported from centralized location


@dataclass
class AnswerCorrectness(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity.

    Attributes
    ----------
    weights:
        List of two weights for factuality and semantic similarity [0.75, 0.25]
    answer_similarity:
        The AnswerSimilarity object
    """

    name: str = "answer_correctness"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference"}
        }
    )
    output_type = MetricOutputType.CONTINUOUS
    weights: list[float] = field(default_factory=lambda: [0.75, 0.25])
    beta: float = 1.0
    answer_similarity: t.Optional[AnswerSimilarity] = None
    max_retries: int = 1

    def __post_init__(self):
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

        if type(self.beta) is not float:
            raise ValueError(
                "Beta must be a float. A beta > 1 gives more weight to recall, while beta < 1 favors precision."
            )

    def init(self, run_config: RunConfig):
        super().init(run_config)
        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(embeddings=self.embeddings)

    def _compute_statement_presence(
        self, prediction: ClassificationWithReason
    ) -> float:
        tp = len(prediction.TP)
        fp = len(prediction.FP)
        fn = len(prediction.FN)
        score = fbeta_score(tp, fp, fn, self.beta)
        return score

    async def _create_simplified_statements(
        self, question: str, text: str
    ) -> StatementGeneratorOutput:
        """Generate statements from text using direct LLM call."""
        assert self.llm is not None, "llm is not set"

        prompt = STATEMENT_GENERATOR_PROMPT.format(question=question, answer=text)

        # Use Instructor LLM interface for direct API calls without LangChain
        result = self.llm.generate(
            prompt,
            response_model=StatementGeneratorOutput,  # type: ignore
        )

        # Instructor returns structured objects directly - no JSON parsing needed!
        return result

    async def _classify_statements(
        self, question: str, answer: list[str], ground_truth: list[str]
    ) -> ClassificationWithReason:
        """Classify statements using direct LLM call."""
        assert self.llm is not None, "llm must be set to compute score"

        answer_json = json.dumps(answer)
        ground_truth_json = json.dumps(ground_truth)

        prompt = CORRECTNESS_CLASSIFIER_PROMPT.format(
            question=question,
            answer_json=answer_json,
            ground_truth_json=ground_truth_json,
        )

        # Use Instructor LLM interface for direct API calls without LangChain
        result = self.llm.generate(
            prompt,
            response_model=ClassificationWithReason,  # type: ignore
        )

        # Instructor returns structured objects directly - no JSON parsing needed!
        return result

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        """Score a single turn sample (callbacks parameter kept for compatibility but ignored)."""
        row = sample.to_dict()
        return await self._ascore(row)

    async def _ascore(self, row: t.Dict, callbacks=None) -> float:
        """
        Calculate answer correctness score.
        """
        assert self.llm is not None, "LLM must be set"

        # extract the statements from the answer and the ground truth
        question = row["user_input"]
        statements: t.Dict[str, t.List[str]] = {}
        for item in ["response", "reference"]:
            statements_x = await self._create_simplified_statements(question, row[item])
            statements[item] = statements_x.statements

        if not all([val == [] for val in statements.values()]):
            ground_truth = [statement for statement in statements["reference"]]
            answer = [statement for statement in statements["response"]]
            answers = await self._classify_statements(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
            )
            if answers is None:
                return np.nan

            f1_score = self._compute_statement_presence(answers)
        else:
            f1_score = 1.0

        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            assert self.answer_similarity is not None, "AnswerSimilarity must be set"

            similarity_score = await self.answer_similarity._ascore(row)

        score = np.average(
            [f1_score, similarity_score],
            weights=self.weights,
        )

        return float(score)


# Create default instance
answer_correctness = AnswerCorrectness()
