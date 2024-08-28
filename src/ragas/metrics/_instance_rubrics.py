from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np

from ragas.llms.prompt import Prompt
from ragas.metrics._rubrics_based import (
    WITH_REFERENCE_SCORING_PROMPT,
    WITHOUT_REFERENCE_SCORING_PROMPT,
    _score_feedback_output_parser,
)
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


@dataclass
class LabelledInstanceRubricsScore(MetricWithLLM):
    name: str = "labelled_rubrics_score"  # type: ignore
    _required_columns: t.Tuple[str, ...] = (
        "user_input",
        "response",
        "reference",
        "rubric",
    )
    evaluation_mode: EvaluationMode = EvaluationMode.qcg  # type: ignore
    scoring_prompt: Prompt = field(
        default_factory=lambda: WITH_REFERENCE_SCORING_PROMPT
    )
    max_retries: int = 1

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_value = self._create_prompt(row)

        response = await self.llm.generate(prompt_value, callbacks=callbacks)

        parsed_response = await _score_feedback_output_parser.aparse(
            response.generations[0][0].text, prompt_value, self.llm, self.max_retries
        )

        if parsed_response is None:
            return np.nan

        score = parsed_response.dicts()[0]["score"]
        return score

    def _create_prompt(self, row: t.Dict) -> PromptValue:
        question, contexts, answer, ground_truth, rubrics = (
            row["user_input"],
            row.get("retrieved_contexts"),
            row["response"],
            row["reference"],
            row["rubric"],
        )
        if contexts is not None:
            contexts = "\n".join(contexts)
            question = f"{question} answer using context: {contexts}"
        return self.scoring_prompt.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            rubrics=rubrics,
        )

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "LLM must be set to adapt the metric"
        self.scoring_prompt.adapt(language, self.llm, cache_dir)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.scoring_prompt.save(cache_dir)


@dataclass
class ReferenceFreeInstanceRubricsScore(LabelledInstanceRubricsScore):
    name: str = "reference_free_rubrics_score"  # type: ignore
    _required_columns: t.Tuple[str, ...] = ("user_input", "response", "rubric")
    evaluation_mode: EvaluationMode = EvaluationMode.qcg  # type: ignore
    scoring_prompt: Prompt = field(
        default_factory=lambda: WITHOUT_REFERENCE_SCORING_PROMPT
    )
    max_retries: int = 1

    def _create_prompt(self, row: t.Dict) -> PromptValue:
        question, contexts, answer, rubrics = (
            row["user_input"],
            row.get("retrieved_contexts"),
            row["response"],
            row["rubric"],
        )
        if contexts is not None:
            contexts = "\n".join(contexts)
            question = f"{question} answer using context: {contexts}"
        return self.scoring_prompt.format(
            question=question,
            answer=answer,
            rubrics=rubrics,
        )


labelled_instancewise_rubrics_score = LabelledInstanceRubricsScore()
reference_instancewise_rubrics_free_rubrics_score = ReferenceFreeInstanceRubricsScore()
