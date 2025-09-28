from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness import (
    STATEMENT_GENERATOR_PROMPT,
    StatementGeneratorOutput,
)
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.metrics.utils import fbeta_score
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


# ============================================================================
# REMOVED LANGCHAIN DEPENDENCIES
# ============================================================================
# The old PydanticPrompt classes (CorrectnessClassifier) have been removed
# to eliminate LangChain dependencies.
#
# If other metrics need these classes, they should be migrated to use the direct
# prompt templates below or create their own LangChain-free implementations.


# ============================================================================
# DIRECT PROMPT TEMPLATES (No PydanticPrompt dependencies)
# ============================================================================

CORRECTNESS_CLASSIFIER_PROMPT = """Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification.

--------EXAMPLES-----------
Example 1
Input: {{"question": "What powers the sun and what is its primary function?", "answer": ["The sun is powered by nuclear fission, similar to nuclear reactors on Earth.", "The primary function of the sun is to provide light to the solar system."], "ground_truth": ["The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "This fusion process in the sun's core releases a tremendous amount of energy.", "The energy from the sun provides heat and light, which are essential for life on Earth.", "The sun's light plays a critical role in Earth's climate system.", "Sunlight helps to drive the weather and ocean currents."]}}
Output: {{"TP": [{{"statement": "The primary function of the sun is to provide light to the solar system.", "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy."}}], "FP": [{{"statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.", "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion."}}], "FN": [{{"statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "reason": "This accurate description of the sun's power source is not included in the answer."}}, {{"statement": "This fusion process in the sun's core releases a tremendous amount of energy.", "reason": "This process and its significance are not mentioned in the answer."}}, {{"statement": "The energy from the sun provides heat and light, which are essential for life on Earth.", "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers."}}, {{"statement": "The sun's light plays a critical role in Earth's climate system.", "reason": "This broader impact of the sun's light on Earth's climate system is not addressed in the answer."}}, {{"statement": "Sunlight helps to drive the weather and ocean currents.", "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer."}}]}}

Example 2
Input: {{"question": "What is the boiling point of water?", "answer": ["The boiling point of water is 100 degrees Celsius at sea level"], "ground_truth": ["The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.", "The boiling point of water can change with altitude."]}}
Output: {{"TP": [{{"statement": "The boiling point of water is 100 degrees Celsius at sea level", "reason": "This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level."}}], "FP": [], "FN": [{{"statement": "The boiling point of water can change with altitude.", "reason": "This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer."}}]}}
-----------------------------

Now perform the same with the following input
input: {{"question": "{question}", "answer": {answer_json}, "ground_truth": {ground_truth_json}}}
Output: """


# ============================================================================
# MIGRATED ANSWER CORRECTNESS METRIC (No LangChain dependencies)
# ============================================================================


@dataclass
class AnswerCorrectness(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Answer Correctness metric without LangChain dependencies.

    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity.

    Key changes from the original implementation:
    - Removed LangChain callback dependencies
    - Uses direct string-based prompts instead of PydanticPrompt classes
    - Simplified LLM interface calls
    - Maintains the same scoring logic and behavior
    - Improved JSON parsing with better error handling

    Attributes
    ----------
    name: string
        The name of the metrics
    weights:
        a list of two weights corresponding to factuality and semantic similarity
        Defaults [0.75, 0.25]
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

        # Use the existing LLM interface but without callbacks
        from langchain_core.prompt_values import StringPromptValue

        prompt_value = StringPromptValue(text=prompt)

        # Generate response using existing LLM interface
        result = await self.llm.generate(prompt_value, n=1, temperature=0.01)

        # Parse JSON response
        response_text = result.generations[0][0].text.strip()
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text

            parsed = json.loads(json_text)
            return StatementGeneratorOutput(statements=parsed.get("statements", []))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse statement generation response: {e}")
            return StatementGeneratorOutput(statements=[])

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

        # Use the existing LLM interface but without callbacks
        from langchain_core.prompt_values import StringPromptValue

        prompt_value = StringPromptValue(text=prompt)

        # Generate response using existing LLM interface
        result = await self.llm.generate(prompt_value, n=1, temperature=0.01)

        # Parse JSON response
        response_text = result.generations[0][0].text.strip()
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text

            parsed = json.loads(json_text)

            # Convert to ClassificationWithReason object
            tp_objects = []
            for stmt_data in parsed.get("TP", []):
                tp_objects.append(
                    StatementsWithReason(
                        statement=stmt_data.get("statement", ""),
                        reason=stmt_data.get("reason", ""),
                    )
                )

            fp_objects = []
            for stmt_data in parsed.get("FP", []):
                fp_objects.append(
                    StatementsWithReason(
                        statement=stmt_data.get("statement", ""),
                        reason=stmt_data.get("reason", ""),
                    )
                )

            fn_objects = []
            for stmt_data in parsed.get("FN", []):
                fn_objects.append(
                    StatementsWithReason(
                        statement=stmt_data.get("statement", ""),
                        reason=stmt_data.get("reason", ""),
                    )
                )

            return ClassificationWithReason(TP=tp_objects, FP=fp_objects, FN=fn_objects)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse classification response: {e}")
            return ClassificationWithReason(TP=[], FP=[], FN=[])

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
