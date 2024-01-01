from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group

from ragas.llms.prompt import Prompt
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.utils import json_loader

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

CORRECTNESS_PROMPT = Prompt(
    name="answer_correctness",
    instruction="""Extract following from given question and ground truth""",
    examples=[
        {
            "question": """What powers the sun and what is its primary function?""",
            "answer": """The sun is powered by nuclear fission, similar to nuclear reactors on Earth, and its primary function is to provide light to the solar system.""",
            "ground_truth": """The sun is actually powered by nuclear fusion, not fission. In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy. This energy is what lights up the sun and provides heat and light, essential for life on Earth. The sun's light also plays a critical role in Earth's climate system and helps to drive the weather and ocean currents.""",
            "Extracted statements": """[
            {
                "statements that are present in both the answer and the ground truth": ["The sun's primary function is to provide light"],
                "statements present in the answer but not found in the ground truth": ["The sun is powered by nuclear fission", "similar to nuclear reactors on Earth"],
                "relevant statements found in the ground truth but omitted in the answer": ["The sun is powered by nuclear fusion, not fission", "In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy", "This energy provides heat and light, essential for life on Earth", "The sun's light plays a critical role in Earth's climate system", "The sun helps to drive the weather and ocean currents"]
            }]
            """,
        },
        {
            "question": """What is the boiling point of water?""",
            "answer": """The boiling point of water is 100 degrees Celsius at sea level.""",
            "ground_truth": """The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level, but it can change with altitude.""",
            "Extracted statements": """[
            {
                "statements that are present in both the answer and the ground truth": ["The boiling point of water is 100 degrees Celsius at sea level"],
                "statements present in the answer but not found in the ground truth": [],
                "relevant statements found in the ground truth but omitted in the answer": ["The boiling point can change with altitude", "The boiling point of water is 212 degrees Fahrenheit at sea level"]
            }]
            """,
        },
    ],
    input_keys=["question", "answer", "ground_truth"],
    output_key="Extracted statements",
    output_type="json",
)


@dataclass
class AnswerCorrectness(MetricWithLLM):

    """
    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity.

    Attributes
    ----------
    name: string
        The name of the metrics
    batch_size: int
        batch size for evaluation
    weights:
        a list of two weights corresponding to factuality and semantic similarity
        Defaults [0.75, 0.25]
    answer_similarity:
        The AnswerSimilarity object
    """

    name: str = "answer_correctness"  # type: ignore[reportIncompatibleMethodOverride]
    evaluation_mode: EvaluationMode = EvaluationMode.qga  # type: ignore[reportIncompatibleMethodOverride]
    batch_size: int = 15
    weights: list[float] = field(default_factory=lambda: [0.75, 0.25])
    answer_similarity: AnswerSimilarity | None = None

    def __post_init__(self: t.Self):
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, batch_size=self.batch_size
            )
        self.correctness_prompt = CORRECTNESS_PROMPT

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        logging.info(f"Adapting AnswerCorrectness metric to {language}")
        self.correctness_prompt = self.correctness_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.correctness_prompt.save(cache_dir)

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[Callbacks] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        question, answer, ground_truths = (
            dataset["question"],
            dataset["answer"],
            dataset["ground_truths"],
        )
        prompts = []

        cb = CallbackManager.configure(inheritable_callbacks=callbacks)
        with trace_as_chain_group(
            callback_group_name, callback_manager=cb
        ) as batch_group:
            for q, a, g in zip(question, answer, ground_truths):
                prompts.append(
                    self.correctness_prompt.format(
                        question=q, ground_truth=g[0], answer=a
                    )
                )

            result = self.llm.generate(prompts, callbacks=batch_group)
            outputs = result.generations
            key_map = {
                "TP": "statements that are present in both the answer and the ground truth",
                "FP": "statements present in the answer but not found in the ground truth",
                "FN": "relevant statements found in the ground truth but omitted in the answer",  # noqa: E501
            }

            f1_score = []
            for prediction in outputs:
                prediction = json_loader.safe_load(prediction[0].text, self.llm)
                prediction = (
                    prediction if isinstance(prediction, list) else [prediction]
                )
                if prediction:
                    prediction = [
                        item.get(key_map[k], np.nan)
                        for item in prediction
                        for k in key_map.keys()
                    ]
                    tp, fp, fn = [
                        len(item) if isinstance(item, list) else np.nan
                        for item in prediction
                    ]
                    score = tp / (tp + 0.5 * (fp + fn))
                else:
                    score = np.nan

                f1_score.append(score)

            if self.weights[1] == 0:
                similarity_scores = np.zeros(len(f1_score))
            else:
                similarity_scores = self.answer_similarity._score_batch(dataset, callbacks=batch_group)  # type: ignore
            scores_stacked = np.vstack([f1_score, similarity_scores])
            scores = np.average(
                scores_stacked,
                axis=0,
                weights=self.weights,
            )

        return scores.tolist()


answer_correctness = AnswerCorrectness()
