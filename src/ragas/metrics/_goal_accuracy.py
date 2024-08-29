from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field

from ragas.dataset_schema import MultiTurnSample
from ragas.llms.output_parser import RagasoutputParser
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks

    from ragas.llms.prompt import PromptValue


class GoalAccuracy(BaseModel):
    user_goal: str = Field(
        ..., description="The task or objective the user wants to achieve."
    )
    end_state: str = Field(
        ..., description="The final outcome or result of the workflow."
    )


class CompareOutcome(BaseModel):
    reason: str = Field(
        ..., description="The task or objective the user wants to achieve."
    )
    verdict: t.Literal["0", "1"] = Field(
        ..., description="The final outcome or result of the workflow."
    )


GOAL_AND_STATE_PROMPT = Prompt(
    name="goal_accuracy",
    instruction="Given an agentic workflow comprised of Human, AI and Tools, identify the user_goal (the task or objective the user wants to achieve) and the end_state (the final outcome or result of the workflow).",
    examples=[
        {
            "workflow": """
            Human: Hey, book a table at the nearest best Chinese restaurant for 8:00pm
            AI: Sure, let me find the best options for you.
            Tools:
                restaurant_search: {'cuisine': 'Chinese', 'time': '8:00pm'}
            ToolOutput: Found a few options: 1. Golden Dragon, 2. Jade Palace
            AI: I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?
            Human: Let's go with Golden Dragon.
            AI: Great choice! I'll book a table for 8:00pm at Golden Dragon.
            Tools:
                restaurant_book: {'name': 'Golden Dragon', 'time': '8:00pm'}
            ToolOutput: Table booked at Golden Dragon for 8:00pm.
            AI: Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!
            Human: thanks
            """,
            "output": {
                "user_goal": "Book a table at the nearest best Chinese restaurant for 8:00pm.",
                "end_state": "A table is successfully booked at Golden Dragon (Chinese restaurant) for 8:00pm.",
            },
        }
    ],
    input_keys=["workflow"],
    output_key="output",
    language="english",
)


COMPARE_OUTCOME_PROMPT = Prompt(
    name="compare_outcome",
    instruction="Given user goal, desired outcome and acheived outcome compare them and identify if they are the same (1) or different(0).",
    examples=[
        {
            "goal": "Book a table at any Chinese restaurant for 8:00pm.",
            "desired_outcome": "A table is successfully booked at any Chinese restaurant for 8:00pm.",
            "arrived_outcome": "A table is successfully booked at Jade Palace (Chinese restaurant) for 8:00pm.",
            "output": {
                "reason": "The arrived outcome is same as the desired outcome and aligns with the user goal.",
                "verdict": "1",
            },
        }
    ],
    input_keys=["goal", "desired_outcome", "arrived_outcome"],
    output_key="output",
    language="english",
)


_goal_accuracy_output_parser = RagasoutputParser(pydantic_object=GoalAccuracy)
_outcome_comparison_parser = RagasoutputParser(pydantic_object=CompareOutcome)


@dataclass
class AgentGoalAccuracy(MetricWithLLM):
    name: str = "agent_goal_accuracy"  # type: ignore
    _required_columns: t.Tuple[str, ...] = ("user_input", "reference")  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qac  # type: ignore
    workflow_prompt: Prompt = field(default_factory=lambda: GOAL_AND_STATE_PROMPT)
    compare_outcome_prompt: Prompt = field(
        default_factory=lambda: COMPARE_OUTCOME_PROMPT
    )
    max_retries: int = 1

    def _create_workflow_prompt(self, sample: MultiTurnSample) -> PromptValue:
        workflow = sample.pretty_repr()
        return self.workflow_prompt.format(workflow=workflow)

    def _create_comparison_prompt(self, row: t.Dict[str, str]) -> PromptValue:
        return self.compare_outcome_prompt.format(
            goal=row["goal"],
            desired_outcome=row["desired_outcome"],
            arrived_outcome=row["arrived_outcome"],
        )

    async def _ascore(
        self,
        row: MultiTurnSample,
        callbacks: Callbacks,
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        prompt_value = self._create_workflow_prompt(row)
        response = await self.llm.generate(prompt_value, callbacks=callbacks)
        response = response.generations[0][0].text
        parsed_response = await _goal_accuracy_output_parser.aparse(
            response, self.llm, self.max_retries
        )
        if parsed_response is None:
            return np.nan

        row_dict = {
            "goal": parsed_response.user_goal,
            "arrived_outcome": parsed_response.end_state,
            "desired_outcome": row.reference,
        }

        prompt = self._create_comparison_prompt(row_dict)
        result = await self.llm.generate(prompt, callbacks=callbacks)
        result = result.generations[0][0].text
        parsed_response = await _outcome_comparison_parser.aparse(
            result, self.llm, self.max_retries
        )
        if parsed_response is None:
            return np.nan
        verdict = int(parsed_response.verdict)
        return verdict


agent_goal_accuracy = AgentGoalAccuracy()
