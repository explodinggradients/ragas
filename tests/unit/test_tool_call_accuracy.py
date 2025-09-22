"""Tests for ToolCallAccuracy metric."""

from unittest.mock import AsyncMock

import pytest

from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage, ToolCall
from ragas.metrics import ToolCallAccuracy


@pytest.fixture
def tool_call_accuracy():
    """Fixture providing ToolCallAccuracy instance."""
    return ToolCallAccuracy()


@pytest.fixture
def mock_callbacks():
    """Fixture providing mock callbacks."""
    return AsyncMock()


class TestToolCallAccuracy:
    """Test cases for ToolCallAccuracy metric."""

    def test_is_sequence_aligned_perfect_match(self, tool_call_accuracy):
        """Test sequence alignment with perfect match."""
        pred_seq = ["func1", "func2", "func3"]
        ref_seq = ["func1", "func2", "func3"]
        assert tool_call_accuracy.is_sequence_aligned(pred_seq, ref_seq) is True

    def test_is_sequence_aligned_different_order(self, tool_call_accuracy):
        """Test sequence alignment with different order."""
        pred_seq = ["func1", "func3", "func2"]
        ref_seq = ["func1", "func2", "func3"]
        assert tool_call_accuracy.is_sequence_aligned(pred_seq, ref_seq) is False

    def test_is_sequence_aligned_different_length(self, tool_call_accuracy):
        """Test sequence alignment with different lengths."""
        pred_seq = ["func1", "func2"]
        ref_seq = ["func1", "func2", "func3"]
        assert tool_call_accuracy.is_sequence_aligned(pred_seq, ref_seq) is False

    def test_is_sequence_aligned_empty_sequences(self, tool_call_accuracy):
        """Test sequence alignment with empty sequences."""
        assert tool_call_accuracy.is_sequence_aligned([], []) is True

    @pytest.mark.asyncio
    async def test_perfect_match_scenario(self, tool_call_accuracy, mock_callbacks):
        """Test perfect match scenario with identical tool calls."""
        # Create reference tool calls
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        # Create predicted tool calls
        pred_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        # Create sample
        sample = MultiTurnSample(
            user_input=[
                AIMessage(content="I'll search for you", tool_calls=pred_tool_calls)
            ],
            reference_tool_calls=ref_tool_calls,
        )

        # Mock the arg comparison to return 1.0 for perfect matches
        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_no_predicted_tool_calls(self, tool_call_accuracy, mock_callbacks):
        """Test case with no predicted tool calls."""
        ref_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="No tool calls here")],
            reference_tool_calls=ref_tool_calls,
        )

        with pytest.warns(UserWarning, match="No tool calls found"):
            score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_sequence_misalignment(self, tool_call_accuracy, mock_callbacks):
        """Test case where sequences don't align."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        # Different order - should result in score 0 due to sequence misalignment
        pred_tool_calls = [
            ToolCall(name="filter", args={"type": "recent"}),
            ToolCall(name="search", args={"query": "python"}),
        ]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_length_mismatch_more_predicted(
        self, tool_call_accuracy, mock_callbacks
    ):
        """Test case with more predicted tool calls than reference."""
        ref_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        pred_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        with pytest.warns(UserWarning, match="Length mismatch"):
            score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)

        # Should be 0 because sequences don't align (different lengths)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_length_mismatch_fewer_predicted(
        self, tool_call_accuracy, mock_callbacks
    ):
        """Test case with fewer predicted tool calls than reference."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        pred_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        with pytest.warns(UserWarning, match="Length mismatch"):
            score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)

        # Should be 0 because sequences don't align (different lengths)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_partial_argument_match(self, tool_call_accuracy, mock_callbacks):
        """Test case with partial argument matches."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python", "limit": 10}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        pred_tool_calls = [
            ToolCall(
                name="search", args={"query": "python", "limit": 5}
            ),  # Wrong limit
            ToolCall(name="filter", args={"type": "recent"}),  # Perfect match
        ]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        # Mock to return scores based on the argument comparison
        # For the "search" tool call: we need to call for each argument
        # For "python" vs "python": 1.0, for 5 vs 10: 0.0 -> average = 0.5
        # For the "filter" tool call: "recent" vs "recent": 1.0 -> average = 1.0
        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            side_effect=[1.0, 0.0, 1.0]  # query match, limit mismatch, type match
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 0.75  # (0.5 + 1.0) / 2

    @pytest.mark.asyncio
    async def test_wrong_tool_names(self, tool_call_accuracy, mock_callbacks):
        """Test case with wrong tool names."""
        ref_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        pred_tool_calls = [ToolCall(name="wrong_tool", args={"query": "python"})]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 0.0  # Wrong tool name should result in 0

    @pytest.mark.asyncio
    async def test_multiple_ai_messages(self, tool_call_accuracy, mock_callbacks):
        """Test case with multiple AI messages containing tool calls."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        # Tool calls spread across multiple messages
        sample = MultiTurnSample(
            user_input=[
                AIMessage(
                    content="First",
                    tool_calls=[ToolCall(name="search", args={"query": "python"})],
                ),
                AIMessage(
                    content="Second",
                    tool_calls=[ToolCall(name="filter", args={"type": "recent"})],
                ),
            ],
            reference_tool_calls=ref_tool_calls,
        )

        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_empty_reference_tool_calls(self, tool_call_accuracy, mock_callbacks):
        """Test case with empty reference tool calls and no predictions."""
        sample = MultiTurnSample(
            user_input=[AIMessage(content="No tools needed")],
            reference_tool_calls=[],
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 1.0  # Both empty should be perfect match

    @pytest.mark.asyncio
    async def test_empty_reference_with_predictions(
        self, tool_call_accuracy, mock_callbacks
    ):
        """Test case with empty reference but predictions exist."""
        sample = MultiTurnSample(
            user_input=[
                AIMessage(
                    content="Calling tool",
                    tool_calls=[ToolCall(name="unexpected", args={})],
                )
            ],
            reference_tool_calls=[],
        )

        with pytest.warns(UserWarning, match="Reference tool calls are empty"):
            score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 0.0

    def test_metric_name(self, tool_call_accuracy):
        """Test that metric has correct name."""
        assert tool_call_accuracy.name == "tool_call_accuracy"

    def test_required_columns(self, tool_call_accuracy):
        """Test that metric has correct required columns."""
        from ragas.metrics.base import MetricType

        required = tool_call_accuracy._required_columns[MetricType.MULTI_TURN]
        assert "user_input" in required
        assert "reference_tool_calls" in required
