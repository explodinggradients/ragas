from __future__ import annotations

import typing as t
import warnings
from dataclasses import dataclass, field

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.messages import AIMessage
from ragas.metrics._string import ExactMatch
from ragas.metrics.base import MetricType, MultiTurnMetric, SingleTurnMetric

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


@dataclass
class ToolCallAccuracy(MultiTurnMetric):
    """
    Tool Call Accuracy metric measures how accurately an LLM agent makes tool calls
    compared to reference tool calls.

    The metric evaluates two aspects:
    1. Sequence alignment: Whether predicted and reference tool calls match exactly in order
    2. Argument accuracy: How well tool call arguments match between predicted and reference

    Score calculation:
    - If sequences don't align exactly: score = 0
    - If sequences align: score = (average argument accuracy) * sequence_alignment_factor
    - Length mismatches result in warnings and proportional penalty

    Edge cases:
    - No predicted tool calls: returns 0.0
    - Length mismatch: compares only the overlapping portion and applies coverage penalty
    - Missing arguments: contributes 0 to the argument score for that tool call

    The final score is always between 0.0 and 1.0.
    """

    name: str = "tool_call_accuracy"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.MULTI_TURN: {
                "user_input",
                "reference_tool_calls",
            }
        }
    )

    arg_comparison_metric: SingleTurnMetric = field(
        default_factory=lambda: ExactMatch()
    )

    def init(self, run_config):
        pass

    async def _get_arg_score(
            self, preds: t.Dict[str, t.Any], refs: t.Dict[str, t.Any], callbacks: Callbacks
    ) -> float:
        if not refs and not preds:
            return 1.0
        if not refs:
            return 0.0

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

    def is_sequence_aligned(
            self, pred_sequence: t.List[str], ref_sequence: t.List[str]
    ) -> bool:
        return pred_sequence == ref_sequence

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        assert sample.reference_tool_calls is not None, (
            "Reference tool calls is not set"
        )

        pred_tool_calls = []
        for item in sample.user_input:
            if isinstance(item, AIMessage) and item.tool_calls is not None:
                pred_tool_calls.extend(item.tool_calls)

        reference_tool_calls = sample.reference_tool_calls

        # Handle edge cases
        if not pred_tool_calls and not reference_tool_calls:
            # Both empty - perfect match
            return 1.0
        elif not pred_tool_calls:
            warnings.warn("No tool calls found in the user input")
            return 0.0
        elif not reference_tool_calls:
            # Reference is empty but we have predictions - this is typically an error in test data
            warnings.warn("Reference tool calls are empty but predictions exist")
            return 0.0

        # Check for length mismatch and warn user
        if len(pred_tool_calls) != len(reference_tool_calls):
            warnings.warn(
                f"Length mismatch: predicted tool calls ({len(pred_tool_calls)}) "
                f"vs reference tool calls ({len(reference_tool_calls)}). "
                f"Only the first {min(len(pred_tool_calls), len(reference_tool_calls))} "
                f"tool calls will be compared."
            )

        tool_call_pred_sequence = [tool_call.name for tool_call in pred_tool_calls]
        tool_call_ref_sequence = [tool_call.name for tool_call in reference_tool_calls]

        sequence_aligned = int(
            self.is_sequence_aligned(tool_call_pred_sequence, tool_call_ref_sequence)
        )

        # Calculate score based on paired tool calls (without nested loop)
        score = 0.0
        compared_count = min(len(pred_tool_calls), len(reference_tool_calls))

        for ref_tool_call, pred_tool_call in zip(reference_tool_calls, pred_tool_calls):
            if ref_tool_call.name == pred_tool_call.name:
                arg_score = await self._get_arg_score(
                    pred_tool_call.args, ref_tool_call.args, callbacks
                )
                score += arg_score

        score /= len(reference_tool_calls)

        if compared_count < len(reference_tool_calls):
            coverage_penalty = compared_count / len(reference_tool_calls)
            score *= coverage_penalty

        return score * sequence_aligned

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._multi_turn_ascore(MultiTurnSample(**row), callbacks)
