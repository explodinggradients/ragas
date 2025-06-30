from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from ragas.metrics.base import MultiTurnMetric, MetricType
from ragas.types import MultiTurnSample
from ragas.utils.typing import ScoreType

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


@dataclass
class ToolCallF1(MultiTurnMetric):
    name: str = "tool_call_f1"
    batch_size: int = 1
    is_multi_turn: bool = True
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.MULTI_TURN: {
                "reference_tool_calls",
                "agent_messages",
            }
        }
    )

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: t.Optional[Callbacks] = None
    ) -> ScoreType:
        expected: set[tuple[str, frozenset]] = set()
        if sample.reference_tool_calls:
            for call in sample.reference_tool_calls:
                expected.add((call.name, frozenset(call.parameters.items())))

        actual: set[tuple[str, frozenset]] = set()
        for msg in sample.agent_messages:
            if msg.tool_calls:
                for call in msg.tool_calls:
                    actual.add((call.name, frozenset(call.parameters.items())))

        tp = len(actual & expected)
        fp = len(actual - expected)
        fn = len(expected - actual)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return round(f1, 4)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> ScoreType:
        return await self._multi_turn_ascore(MultiTurnSample(**row), callbacks)
