from __future__ import annotations

import typing as t
from enum import Enum
import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


class PrometheusMode(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class ScoreFeedback(BaseModel):
    feedback: str = Field(..., description="The feedback for the response")
    score: int = Field(..., description="The score given to the response")


class ScoreFeedbackAnswers(BaseModel):
    __root__: t.List[ScoreFeedback]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_score_feedback_output_instructions = get_json_format_instructions(ScoreFeedbackAnswers)
_score_feedback_output_parser = RagasoutputParser(pydantic_object=ScoreFeedbackAnswers)


PROMETHEUS_ABSOLUTE_PROMPT = Prompt(
    name="prometheus_eval",
    instruction="Evaluate the response based on the given question, reference answer, and rubric. Provide a 'feedback' and 'score' out of 5 in JSON format.",
    examples=[
        {
            "question": "Explain the concept of recursion in programming.",
            "answer": "Recursion is when a function calls itself.",
            "ground_truth": "Recursion is a programming technique where a function calls itself to solve smaller instances of a problem, often with a base case to prevent infinite loops.",
            "rubrics": {
                "criteria": "Does the answer accurately explain recursion and its key components?",
                "score1_description": "Fails to capture the essence of recursion, offering an incomplete or incorrect explanation.",
                "score2_description": "Provides a very basic explanation of recursion but misses key components, such as the base case.",
                "score3_description": "Correctly explains recursion but lacks depth, particularly in discussing the base case and its importance.",
                "score4_description": "Gives a mostly accurate and detailed explanation of recursion, including the base case, with minor omissions.",
                "score5_description": "Provides a clear and complete explanation of recursion, including a thorough discussion of the base case and its significance."
            },
            "analysis": {"feedback": "The answer is correct but lacks details on the base case, which is crucial for understanding recursion fully.", "score": 3}
        }
    ],
    input_keys=["question", "answer", "ground_truth", "rubrics"],
    output_key="analysis",
    language="english",
)


class Prometheus(MetricWithLLM):
    name = "prometheus"
    evaluation_mode = EvaluationMode.qga

    def __init__(
            self,
            mode: PrometheusMode = PrometheusMode.ABSOLUTE,
            rubrics: Optional[Dict] = None,
            llm: Optional[BaseRagasLLM] = None,
            max_retries: int = 1,
    ):
        super().__init__(llm=llm)
        self.mode = mode
        self.rubrics = rubrics
        self.max_retries = max_retries


    async def _ascore(self, row: Dict, callbacks: t.Callbacks, is_async: bool = False) -> float:
        if self.mode == PrometheusMode.ABSOLUTE:
            return await self._absolute_score(row, callbacks)
        elif self.mode == PrometheusMode.RELATIVE:
            return await self._relative_score(row, callbacks)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _create_prompt(self, row: Dict) -> Prompt:
        return PROMETHEUS_ABSOLUTE_PROMPT.format(
            question=row.get('question', ''),
            answer=row.get('answer', ''),
            ground_truth=row.get('ground_truth', ''),
            rubrics=self.rubrics,
        )

    async def _absolute_score(self, row: Dict, callbacks: t.Callbacks) -> float:
        prompt_value = self._create_prompt(row)


        response = await self.llm.generate(prompt_value, callbacks=callbacks)


        parsed_response = await _score_feedback_output_parser.aparse(
            response.generations[0][0].text, prompt_value, self.llm, self.max_retries
        )

        if parsed_response is None:
            return np.nan

        score = parsed_response.dicts()[0]['score']
        return score

    async def _relative_score(self, row: Dict, callbacks: t.Callbacks) -> float:
        # Implement relative scoring logic here, similar to absolute scoring
        return 0.5

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        PROMETHEUS_ABSOLUTE_PROMPT.adapt(language, self.llm, cache_dir)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        PROMETHEUS_ABSOLUTE_PROMPT.save(cache_dir)
