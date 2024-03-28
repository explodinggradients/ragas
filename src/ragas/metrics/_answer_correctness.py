from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics.base import EvaluationMode, MetricWithEmbeddings, MetricWithLLM
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


logger = logging.getLogger(__name__)


class AnswerCorrectnessClassification(BaseModel):
    TP: t.List[str]
    FP: t.List[str]
    FN: t.List[str]


_output_instructions = get_json_format_instructions(AnswerCorrectnessClassification)
_output_parser = RagasoutputParser(pydantic_object=AnswerCorrectnessClassification)

CORRECTNESS_INSTRUCTIONS = """\
Given a ground truth and an answer, analyze each statement in the answer and classify them in one of the following categories:

- TP (true positive): statements that are present in both the answer and the ground truth,
- FP (false positive): statements present in the answer but not found in the ground truth,
- FN (false negative): relevant statements found in the ground truth but omitted in the answer.

A single statement you must classify in exactly one category. Do not try to interpret the meaning of the ground truth or the answer, just compare the presence of the statements in them."""

CORRECTNESS_PROMPT = Prompt(
    name="answer_correctness",
    instruction=CORRECTNESS_INSTRUCTIONS,
    output_format_instruction=_output_instructions,
    examples=[
        {
            "question": """What powers the sun and what is its primary function?""",
            "answer": """The sun is powered by nuclear fission, similar to nuclear reactors on Earth, and its primary function is to provide light to the solar system.""",
            "ground_truth": """The sun is actually powered by nuclear fusion, not fission. In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy. This energy is what lights up the sun and provides heat and light, essential for life on Earth. The sun's light also plays a critical role in Earth's climate system and helps to drive the weather and ocean currents.""",
            "extracted_statements": AnswerCorrectnessClassification.parse_obj(
                {
                    "TP": ["The sun's primary function is to provide light"],
                    "FP": [
                        "The sun is powered by nuclear fission",
                        "similar to nuclear reactors on Earth",
                    ],
                    "FN": [
                        "The sun is powered by nuclear fusion, not fission",
                        "In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy",
                        "This energy provides heat and light, essential for life on Earth",
                        "The sun's light plays a critical role in Earth's climate system",
                        "The sun helps to drive the weather and ocean currents",
                    ],
                }
            ).dict(),
        },
        {
            "question": """What is the boiling point of water?""",
            "answer": """The boiling point of water is 100 degrees Celsius at sea level.""",
            "ground_truth": """The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level, but it can change with altitude.""",
            "extracted_statements": AnswerCorrectnessClassification.parse_obj(
                {
                    "TP": [
                        "The boiling point of water is 100 degrees Celsius at sea level"
                    ],
                    "FP": [],
                    "FN": [
                        "The boiling point can change with altitude",
                        "The boiling point of water is 212 degrees Fahrenheit at sea level",
                    ],
                }
            ).dict(),
        },
    ],
    input_keys=["question", "answer", "ground_truth"],
    output_key="extracted_statements",
    output_type="json",
)


@dataclass
class AnswerCorrectness(MetricWithLLM, MetricWithEmbeddings):

    """
    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity.

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

    name: str = "answer_correctness"  # type: ignore[reportIncompatibleMethodOverride]
    evaluation_mode: EvaluationMode = EvaluationMode.qga  # type: ignore[reportIncompatibleMethodOverride]
    correctness_prompt: Prompt = field(default_factory=lambda: CORRECTNESS_PROMPT)
    weights: list[float] = field(default_factory=lambda: [0.75, 0.25])
    answer_similarity: AnswerSimilarity | None = None
    max_retries: int = 1

    def __post_init__(self: t.Self):
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

    def init(self, run_config: RunConfig):
        super().init(run_config)
        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, embeddings=self.embeddings
            )

    def _compute_statement_presence(
        self, prediction: AnswerCorrectnessClassification
    ) -> float:
        tp = len(prediction.TP)
        fp = len(prediction.FP)
        fn = len(prediction.FN)
        score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0
        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks, is_async: bool) -> float:
        assert self.llm is not None, "LLM must be set"

        q, a, g = row["question"], row["answer"], row["ground_truth"]
        p_value = self.correctness_prompt.format(question=q, ground_truth=g, answer=a)
        is_statement_present = await self.llm.generate(
            p_value, callbacks=callbacks, is_async=is_async
        )
        result_text = is_statement_present.generations[0][0].text

        answers = await _output_parser.aparse(
            result_text, p_value, self.llm, self.max_retries
        )
        if answers is None:
            return np.nan

        f1_score = self._compute_statement_presence(answers)

        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            assert self.answer_similarity is not None, "AnswerSimilarity must be set"

            similarity_score = await self.answer_similarity.ascore(
                row, callbacks=callbacks, is_async=is_async
            )

        score = np.average(
            [f1_score, similarity_score],
            weights=self.weights,
        )

        return float(score)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "llm must be set to compute score"

        logger.info(f"Adapting AnswerCorrectness metric to {language}")
        self.correctness_prompt = self.correctness_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.correctness_prompt.save(cache_dir)


answer_correctness = AnswerCorrectness()
