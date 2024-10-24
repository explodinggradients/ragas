import typing as t
from dataclasses import dataclass, field
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
from pydantic import BaseModel, Field
from ragas.prompt import ImageTextPrompt
from ragas.dataset_schema import SingleTurnSample
import numpy as np


class FaithfulnessInput(BaseModel):
    response: str = Field(description="response from AI")
    retrieved_contexts: list[str] = Field(description="contexts retrieved from the LLM")

    def to_string_list(self):
        return [
            "inputs:",
            self.response,
            "retrieved_contexts: ",
        ] + self.retrieved_contexts


class FaithfulnessOutput(BaseModel):
    faithful: bool = Field(description="boolean indicating if request was faithful")


class MultiModalFaithfulnessPrompt(
    ImageTextPrompt[FaithfulnessInput, FaithfulnessOutput]
):
    # refer: https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/evaluation/multi_modal/faithfulness.py
    instruction = "Please tell if a given piece of information is supported by the visual as well as textual context information. You need to answer with either True or False. Answer True if any of the image(s) and textual context supports the information"
    input_model = FaithfulnessInput
    output_model = FaithfulnessOutput
    examples = [
        (
            FaithfulnessInput(
                response="Apple pie is generally double-crusted.",
                retrieved_contexts=[
                    "An apple pie is a fruit pie in which the principal filling ingredient is apples.",
                    "Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.",
                    "It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).",
                ],
            ),
            FaithfulnessOutput(faithful=True),
        ),
        (
            FaithfulnessInput(
                response="Apple pies tastes bad.",
                retrieved_contexts=[
                    "An apple pie is a fruit pie in which the principal filling ingredient is apples.",
                    "Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.",
                    "It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).",
                ],
            ),
            FaithfulnessOutput(faithful=False),
        ),
    ]


@dataclass
class MultiModalFaithfulness(MetricWithLLM, SingleTurnMetric):
    name: str = "faithful_rate"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "response",
                "retrieved_contexts",
            }
        }
    )
    faithfulness_prompt: ImageTextPrompt = MultiModalFaithfulnessPrompt()

    async def _ascore(self, row: t.Dict, callbacks: t.Any) -> float:
        prompt_input = FaithfulnessInput(
            response=row["response"], retrieved_contexts=row["retrieved_contexts"]
        )
        assert self.llm is not None, "LLM is not set"
        prompt_response = await self.faithfulness_prompt.generate(
            data=prompt_input, llm=self.llm
        )
        if prompt_response is None:
            return np.nan
        return float(prompt_response.faithful)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: t.Any
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)


multimodal_faithness = MultiModalFaithfulness()
