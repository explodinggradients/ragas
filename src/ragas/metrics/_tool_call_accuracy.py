from __future__ import annotations

import typing as t
import warnings
from dataclasses import dataclass, field

import numpy as np

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.messages import AIMessage
from ragas.metrics._string import ExactMatch
from ragas.metrics.base import MetricType, MultiTurnMetric, SingleTurnMetric

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


@dataclass
class ToolCallAccuracy(MultiTurnMetric):
    name: str = "tool_call_accuracy"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.MULTI_TURN: {
                "user_input",
                "reference",
            }
        }
    )

    arg_comparison_metric: SingleTurnMetric = ExactMatch()

    def init(self, run_config):
        pass

    async def _get_arg_score(
        self, preds: t.Dict[str, t.Any], refs: t.Dict[str, t.Any], callbacks: Callbacks
    ) -> float:
        score = 0.0
        for arg in refs.keys():
            if arg in preds:
                score += await self.arg_comparison_metric.single_turn_ascore(
                    SingleTurnSample(
                        response=str(preds[arg]), reference=str(refs[arg])
                    ),
                    callbacks,
                )

        return score / len(refs.keys())

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        assert sample.reference_tool_calls is not None, "Reference is not set"

        if isinstance(sample.user_input[-1], AIMessage):
            if sample.user_input[-1].tool_calls is None:
                return np.nan

            score = 0.0
            reference_tool_calls = sample.reference_tool_calls
            for ref_tool_call in reference_tool_calls:
                for pred_tool_call in sample.user_input[-1].tool_calls:
                    if ref_tool_call.name == pred_tool_call.name:
                        arg_score = await self._get_arg_score(
                            pred_tool_call.args, ref_tool_call.args, callbacks
                        )
                        score += arg_score

            return score / len(reference_tool_calls)
        else:
            warnings.warn("Last message is not an AIMessage with ToolCalls")
            return np.nan

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._multi_turn_ascore(MultiTurnSample(**row), callbacks)
