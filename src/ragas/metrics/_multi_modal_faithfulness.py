import typing as t
from dataclasses import dataclass, field
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
from pydantic import BaseModel, Field
from ragas.prompt import ImageTextPrompt


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
    name: str = "faithful_rate"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "response",
                "retrieved_contexts",
            }
        }
    )
    faithfulness_prompt: ImageTextPrompt = MultiModalFaithfulnessPrompt()

    async def _ascore(self, row):
        pass

    async def _single_turn_ascore(self, sample, callbacks):
        prompt_input = FaithfulnessInput(
            response=sample.response, retrieved_contexts=sample.retrieved_contexts
        )
        prompt_response = await self.faithfulness_prompt.generate(
            data=prompt_input, llm=self.llm
        )
        return prompt_response.faithful


multimodal_faithness = MultiModalFaithfulness()
