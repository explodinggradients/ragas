from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.utils import json_loader

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

CORRECTNESS_PROMPT = HumanMessagePromptTemplate.from_template(
    """
Extract following from given question and ground truth

Question:What powers the sun and what is its primary function?
Answer: The sun is powered by nuclear fission, similar to nuclear reactors on Earth, and its primary function is to provide light to the solar system.
Ground truth: The sun is actually powered by nuclear fusion, not fission. In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy. This energy is what lights up the sun and provides heat and light, essential for life on Earth. The sun's light also plays a critical role in Earth's climate system and helps to drive the weather and ocean currents.
Extracted statements:
[
{{
  "statements that are present in both the answer and the ground truth": ["The sun's primary function is to provide light"],
  "statements present in the answer but not found in the ground truth": ["The sun is powered by nuclear fission", "similar to nuclear reactors on Earth"],
  "relevant statements found in the ground truth but omitted in the answer": ["The sun is powered by nuclear fusion, not fission", "In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy", "This energy provides heat and light, essential for life on Earth", "The sun's light plays a critical role in Earth's climate system", "The sun helps to drive the weather and ocean currents"]
}}
]

Question: What is the boiling point of water?
Answer: The boiling point of water is 100 degrees Celsius at sea level.
Ground truth: The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level, but it can change with altitude.
Extracted statements:
[
  {{
    "statements that are present in both the answer and the ground truth": ["The boiling point of water is 100 degrees Celsius at sea level"],
    "statements present in the answer but not found in the ground truth": [],
    "relevant statements found in the ground truth but omitted in the answer": ["The boiling point can change with altitude", "The boiling point of water is 212 degrees Fahrenheit at sea level"]
  }}
]


Question:{question}
Answer: {answer}
Ground truth: {ground_truth}
Extracted statements:"""  # noqa: E501
)


@dataclass
class AnswerCorrectness(MetricWithLLM):

    """
    Measures answer correctness compared to ground truth as a combination of
    semantic similarity and factuality

    Attributes
    ----------
    name: string
        The name of the metrics
    batch_size: int
        batch size for evaluation
    weights:
        a list of two weights corresponding to semantic similarity and factuality
        Defaults [0.5, 0.5]
    answer_similarity:
        The AnswerSimilarity object
    faithfulness
        The faithfulness object
    """

    name: str = "answer_correctness"  # type: ignore[reportIncompatibleMethodOverride]
    evaluation_mode: EvaluationMode = EvaluationMode.qga  # type: ignore[reportIncompatibleMethodOverride]
    batch_size: int = 15
    weights: list[float] = field(default_factory=lambda: [0.75, 0.25])
    answer_similarity: AnswerSimilarity | None = None

    def __post_init__(self: t.Self):
        if self.answer_similarity is None:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, batch_size=self.batch_size
            )

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
                human_prompt = CORRECTNESS_PROMPT.format(
                    question=q, ground_truth=g[0], answer=a
                )
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

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
            prediction = prediction if isinstance(prediction, list) else []
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

        similarity_scores = self.answer_similarity._score_batch(dataset)  # type: ignore
        scores_stacked = np.vstack([f1_score, similarity_scores])
        scores = np.average(
            scores_stacked,
            axis=0,
            weights=self.weights,
        )

        return scores.tolist()


answer_correctness = AnswerCorrectness()
