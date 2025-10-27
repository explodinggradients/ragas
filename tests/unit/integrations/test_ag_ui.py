"""Tests for AG-UI integration."""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import patch

import pytest

from ragas.messages import AIMessage, HumanMessage, ToolMessage


# Mock AG-UI types for testing without requiring ag-ui-protocol installation
class MockEventType:
    """Mock EventType enum."""

    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"


class MockEvent:
    """Base mock event."""

    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        self.timestamp = kwargs.get("timestamp", 1234567890)
        self.raw_event = kwargs.get("raw_event")
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockRunStartedEvent(MockEvent):
    """Mock RunStartedEvent."""

    def __init__(self, run_id: str, thread_id: str, **kwargs):
        super().__init__(MockEventType.RUN_STARTED, **kwargs)
        self.run_id = run_id
        self.thread_id = thread_id


class MockStepStartedEvent(MockEvent):
    """Mock StepStartedEvent."""

    def __init__(self, step_name: str, **kwargs):
        super().__init__(MockEventType.STEP_STARTED, **kwargs)
        self.step_name = step_name


class MockStepFinishedEvent(MockEvent):
    """Mock StepFinishedEvent."""

    def __init__(self, step_name: str, **kwargs):
        super().__init__(MockEventType.STEP_FINISHED, **kwargs)
        self.step_name = step_name


class MockTextMessageStartEvent(MockEvent):
    """Mock TextMessageStartEvent."""

    def __init__(self, message_id: str, role: str = "assistant", **kwargs):
        super().__init__(MockEventType.TEXT_MESSAGE_START, **kwargs)
        self.message_id = message_id
        self.role = role


class MockTextMessageContentEvent(MockEvent):
    """Mock TextMessageContentEvent."""

    def __init__(self, message_id: str, delta: str, **kwargs):
        super().__init__(MockEventType.TEXT_MESSAGE_CONTENT, **kwargs)
        self.message_id = message_id
        self.delta = delta


class MockTextMessageEndEvent(MockEvent):
    """Mock TextMessageEndEvent."""

    def __init__(self, message_id: str, **kwargs):
        super().__init__(MockEventType.TEXT_MESSAGE_END, **kwargs)
        self.message_id = message_id


class MockToolCallStartEvent(MockEvent):
    """Mock ToolCallStartEvent."""

    def __init__(
        self,
        tool_call_id: str,
        tool_call_name: str,
        parent_message_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(MockEventType.TOOL_CALL_START, **kwargs)
        self.tool_call_id = tool_call_id
        self.tool_call_name = tool_call_name
        self.parent_message_id = parent_message_id


class MockToolCallArgsEvent(MockEvent):
    """Mock ToolCallArgsEvent."""

    def __init__(self, tool_call_id: str, delta: str, **kwargs):
        super().__init__(MockEventType.TOOL_CALL_ARGS, **kwargs)
        self.tool_call_id = tool_call_id
        self.delta = delta


class MockToolCallEndEvent(MockEvent):
    """Mock ToolCallEndEvent."""

    def __init__(self, tool_call_id: str, **kwargs):
        super().__init__(MockEventType.TOOL_CALL_END, **kwargs)
        self.tool_call_id = tool_call_id


class MockToolCallResultEvent(MockEvent):
    """Mock ToolCallResultEvent."""

    def __init__(
        self,
        tool_call_id: str,
        message_id: str,
        content: str,
        role: str = "tool",
        **kwargs,
    ):
        super().__init__(MockEventType.TOOL_CALL_RESULT, **kwargs)
        self.tool_call_id = tool_call_id
        self.message_id = message_id
        self.content = content
        self.role = role


class MockMessage:
    """Mock AG-UI Message object."""

    def __init__(self, role: str, content: str, id: Optional[str] = None):
        self.role = role
        self.content = content
        self.id = id
        self.tool_calls = None


class MockMessagesSnapshotEvent(MockEvent):
    """Mock MessagesSnapshotEvent."""

    def __init__(self, messages: List[MockMessage], **kwargs):
        super().__init__(MockEventType.MESSAGES_SNAPSHOT, **kwargs)
        self.messages = messages


@pytest.fixture
def mock_ag_ui_imports():
    """Mock AG-UI imports for testing."""
    mock_imports = (
        MockEvent,
        MockEventType,
        MockMessagesSnapshotEvent,
        MockTextMessageStartEvent,
        MockTextMessageContentEvent,
        MockTextMessageEndEvent,
        MockToolCallStartEvent,
        MockToolCallArgsEvent,
        MockToolCallEndEvent,
        MockToolCallResultEvent,
    )

    with patch(
        "ragas.integrations.ag_ui._import_ag_ui_core", return_value=mock_imports
    ):
        yield


@pytest.fixture
def basic_text_message_events():
    """Create a basic streaming text message event sequence."""
    return [
        MockRunStartedEvent(run_id="run-123", thread_id="thread-456"),
        MockTextMessageStartEvent(message_id="msg-1", role="user"),
        MockTextMessageContentEvent(message_id="msg-1", delta="Hello"),
        MockTextMessageContentEvent(message_id="msg-1", delta=" world"),
        MockTextMessageEndEvent(message_id="msg-1"),
        MockTextMessageStartEvent(message_id="msg-2", role="assistant"),
        MockTextMessageContentEvent(message_id="msg-2", delta="Hi"),
        MockTextMessageContentEvent(message_id="msg-2", delta=" there!"),
        MockTextMessageEndEvent(message_id="msg-2"),
    ]


@pytest.fixture
def tool_call_events():
    """Create events with tool calls."""
    return [
        MockTextMessageStartEvent(message_id="msg-1", role="assistant"),
        MockTextMessageContentEvent(
            message_id="msg-1", delta="Let me check the weather"
        ),
        MockTextMessageEndEvent(message_id="msg-1"),
        MockToolCallStartEvent(
            tool_call_id="tc-1", tool_call_name="get_weather", parent_message_id="msg-1"
        ),
        MockToolCallArgsEvent(tool_call_id="tc-1", delta='{"city": "San Francisco"'),
        MockToolCallArgsEvent(tool_call_id="tc-1", delta=', "units": "fahrenheit"}'),
        MockToolCallEndEvent(tool_call_id="tc-1"),
        MockToolCallResultEvent(
            tool_call_id="tc-1",
            message_id="result-1",
            content="Temperature: 72°F, Conditions: Sunny",
        ),
        MockTextMessageStartEvent(message_id="msg-2", role="assistant"),
        MockTextMessageContentEvent(
            message_id="msg-2", delta="It's sunny and 72°F in San Francisco"
        ),
        MockTextMessageEndEvent(message_id="msg-2"),
    ]


def test_import_error_without_ag_ui_protocol():
    """Test that appropriate error is raised without ag-ui-protocol package."""
    # This test verifies the error message in _import_ag_ui_core
    # We need to actually call the import function without mocking it
    # to test the error transformation
    from ragas.integrations.ag_ui import _import_ag_ui_core

    # Mock the actual ag_ui import
    with patch.dict("sys.modules", {"ag_ui": None, "ag_ui.core": None}):
        with pytest.raises(
            ImportError, match="AG-UI integration requires the ag-ui-protocol package"
        ):
            _import_ag_ui_core()


def test_basic_text_message_conversion(mock_ag_ui_imports, basic_text_message_events):
    """Test converting basic streaming text messages."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages(basic_text_message_events)

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello world"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Hi there!"


def test_message_with_metadata(mock_ag_ui_imports, basic_text_message_events):
    """Test that metadata is included when requested."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages(basic_text_message_events, metadata=True)

    assert len(messages) == 2
    assert messages[0].metadata is not None
    assert "message_id" in messages[0].metadata
    assert messages[0].metadata["message_id"] == "msg-1"
    assert "run_id" in messages[0].metadata
    assert messages[0].metadata["run_id"] == "run-123"
    assert "thread_id" in messages[0].metadata
    assert messages[0].metadata["thread_id"] == "thread-456"


def test_message_without_metadata(mock_ag_ui_imports, basic_text_message_events):
    """Test that metadata is excluded when not requested."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages(basic_text_message_events, metadata=False)

    assert len(messages) == 2
    assert messages[0].metadata is None
    assert messages[1].metadata is None


def test_tool_call_conversion(mock_ag_ui_imports, tool_call_events):
    """Test converting tool calls with arguments and results."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages(tool_call_events)

    # Should have: AI message, Tool result, AI message
    assert len(messages) == 3

    # First message: AI initiating tool call
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Let me check the weather"
    # Note: tool calls are accumulated and attached to next AI message
    # due to event order

    # Second message: Tool result
    assert isinstance(messages[1], ToolMessage)
    assert "72°F" in messages[1].content

    # Third message: AI with response
    assert isinstance(messages[2], AIMessage)
    assert "sunny" in messages[2].content.lower()


def test_tool_call_with_metadata(mock_ag_ui_imports, tool_call_events):
    """Test that tool call metadata is preserved."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages(tool_call_events, metadata=True)

    tool_message = next(msg for msg in messages if isinstance(msg, ToolMessage))
    assert tool_message.metadata is not None
    assert "tool_call_id" in tool_message.metadata
    assert tool_message.metadata["tool_call_id"] == "tc-1"


def test_step_context_in_metadata(mock_ag_ui_imports):
    """Test that step context is included in metadata."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        MockRunStartedEvent(run_id="run-1", thread_id="thread-1"),
        MockStepStartedEvent(step_name="analyze_query"),
        MockTextMessageStartEvent(message_id="msg-1", role="assistant"),
        MockTextMessageContentEvent(message_id="msg-1", delta="Processing..."),
        MockTextMessageEndEvent(message_id="msg-1"),
        MockStepFinishedEvent(step_name="analyze_query"),
    ]

    messages = convert_to_ragas_messages(events, metadata=True)

    assert len(messages) == 1
    assert "step_name" in messages[0].metadata
    assert messages[0].metadata["step_name"] == "analyze_query"


def test_messages_snapshot_conversion(mock_ag_ui_imports):
    """Test converting MessagesSnapshotEvent."""
    from ragas.integrations.ag_ui import convert_messages_snapshot

    snapshot = MockMessagesSnapshotEvent(
        messages=[
            MockMessage(role="user", content="What's 2+2?", id="msg-1"),
            MockMessage(role="assistant", content="4", id="msg-2"),
            MockMessage(role="user", content="Thanks!", id="msg-3"),
        ]
    )

    messages = convert_messages_snapshot(snapshot)

    assert len(messages) == 3
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "What's 2+2?"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "4"
    assert isinstance(messages[2], HumanMessage)
    assert messages[2].content == "Thanks!"


def test_snapshot_with_metadata(mock_ag_ui_imports):
    """Test that snapshot conversion includes metadata when requested."""
    from ragas.integrations.ag_ui import convert_messages_snapshot

    snapshot = MockMessagesSnapshotEvent(
        messages=[MockMessage(role="user", content="Hello", id="msg-1")]
    )

    messages = convert_messages_snapshot(snapshot, metadata=True)

    assert messages[0].metadata is not None
    assert "message_id" in messages[0].metadata
    assert messages[0].metadata["message_id"] == "msg-1"


def test_non_message_events_filtered(mock_ag_ui_imports):
    """Test that non-message events are silently filtered."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        MockRunStartedEvent(run_id="run-1", thread_id="thread-1"),
        MockEvent(MockEventType.STATE_SNAPSHOT, snapshot={"key": "value"}),
        MockTextMessageStartEvent(message_id="msg-1", role="assistant"),
        MockTextMessageContentEvent(message_id="msg-1", delta="Hello"),
        MockTextMessageEndEvent(message_id="msg-1"),
        MockEvent("RUN_FINISHED", result="success"),
    ]

    messages = convert_to_ragas_messages(events)

    # Should only have the text message, other events filtered
    assert len(messages) == 1
    assert messages[0].content == "Hello"


def test_incomplete_message_stream(mock_ag_ui_imports, caplog):
    """Test handling of incomplete message streams."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    # Message with content but no end event
    events = [
        MockTextMessageStartEvent(message_id="msg-1", role="assistant"),
        MockTextMessageContentEvent(message_id="msg-1", delta="Hello"),
        # Missing TextMessageEndEvent
    ]

    messages = convert_to_ragas_messages(events)

    # Should not create message without end event
    assert len(messages) == 0


def test_orphaned_content_event(mock_ag_ui_imports, caplog):
    """Test handling of content event without corresponding start."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        # Content event without start
        MockTextMessageContentEvent(message_id="msg-unknown", delta="Orphaned content"),
    ]

    messages = convert_to_ragas_messages(events)

    assert len(messages) == 0
    # Should log warning about unknown message_id


def test_tool_call_argument_parsing_error(mock_ag_ui_imports, caplog):
    """Test handling of invalid JSON in tool arguments."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        MockTextMessageStartEvent(message_id="msg-1", role="assistant"),
        MockTextMessageContentEvent(message_id="msg-1", delta="Using tool"),
        MockTextMessageEndEvent(message_id="msg-1"),
        MockToolCallStartEvent(tool_call_id="tc-1", tool_call_name="broken_tool"),
        MockToolCallArgsEvent(tool_call_id="tc-1", delta="{invalid json"),
        MockToolCallEndEvent(tool_call_id="tc-1"),
    ]

    messages = convert_to_ragas_messages(events)

    # Should still create message, but tool call might have raw_args
    assert len(messages) == 1


def test_event_collector_reuse(mock_ag_ui_imports, basic_text_message_events):
    """Test that AGUIEventCollector can be cleared and reused."""
    from ragas.integrations.ag_ui import AGUIEventCollector

    collector = AGUIEventCollector()

    # Process first batch
    for event in basic_text_message_events[:5]:  # First message
        collector.process_event(event)

    messages1 = collector.get_messages()
    assert len(messages1) == 1

    # Clear and process second batch
    collector.clear()
    for event in basic_text_message_events[5:]:  # Second message
        collector.process_event(event)

    messages2 = collector.get_messages()
    assert len(messages2) == 1
    assert messages2[0].content != messages1[0].content


def test_multiple_tool_calls_in_sequence(mock_ag_ui_imports):
    """Test handling multiple tool calls in sequence."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        MockToolCallStartEvent(tool_call_id="tc-1", tool_call_name="tool1"),
        MockToolCallArgsEvent(tool_call_id="tc-1", delta='{"param": "value1"}'),
        MockToolCallEndEvent(tool_call_id="tc-1"),
        MockToolCallStartEvent(tool_call_id="tc-2", tool_call_name="tool2"),
        MockToolCallArgsEvent(tool_call_id="tc-2", delta='{"param": "value2"}'),
        MockToolCallEndEvent(tool_call_id="tc-2"),
        MockTextMessageStartEvent(message_id="msg-1", role="assistant"),
        MockTextMessageContentEvent(message_id="msg-1", delta="Done"),
        MockTextMessageEndEvent(message_id="msg-1"),
    ]

    messages = convert_to_ragas_messages(events)

    # Should create AI message with both tool calls
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls is not None
    assert len(messages[0].tool_calls) == 2
    assert messages[0].tool_calls[0].name == "tool1"
    assert messages[0].tool_calls[1].name == "tool2"


def test_empty_event_list(mock_ag_ui_imports):
    """Test handling of empty event list."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages([])
    assert len(messages) == 0


def test_wrong_snapshot_type_error(mock_ag_ui_imports):
    """Test that convert_messages_snapshot validates input type."""
    from ragas.integrations.ag_ui import convert_messages_snapshot

    with pytest.raises(TypeError, match="Expected MessagesSnapshotEvent"):
        convert_messages_snapshot(MockEvent("WRONG_TYPE"))


def test_role_mapping(mock_ag_ui_imports):
    """Test that different roles map correctly to Ragas message types."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        MockTextMessageStartEvent(message_id="msg-1", role="user"),
        MockTextMessageContentEvent(message_id="msg-1", delta="User message"),
        MockTextMessageEndEvent(message_id="msg-1"),
        MockTextMessageStartEvent(message_id="msg-2", role="assistant"),
        MockTextMessageContentEvent(message_id="msg-2", delta="Assistant message"),
        MockTextMessageEndEvent(message_id="msg-2"),
    ]

    messages = convert_to_ragas_messages(events)

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)


def test_complex_conversation_flow(mock_ag_ui_imports):
    """Test a complex multi-turn conversation with tool calls."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        MockRunStartedEvent(run_id="run-1", thread_id="thread-1"),
        # User asks
        MockTextMessageStartEvent(message_id="msg-1", role="user"),
        MockTextMessageContentEvent(message_id="msg-1", delta="What's the weather?"),
        MockTextMessageEndEvent(message_id="msg-1"),
        # Assistant responds and calls tool
        MockTextMessageStartEvent(message_id="msg-2", role="assistant"),
        MockTextMessageContentEvent(message_id="msg-2", delta="Let me check"),
        MockTextMessageEndEvent(message_id="msg-2"),
        MockToolCallStartEvent(tool_call_id="tc-1", tool_call_name="weather_api"),
        MockToolCallArgsEvent(tool_call_id="tc-1", delta='{"location": "SF"}'),
        MockToolCallEndEvent(tool_call_id="tc-1"),
        # Tool returns result
        MockToolCallResultEvent(
            tool_call_id="tc-1", message_id="result-1", content="Sunny, 70F"
        ),
        # Assistant responds with answer
        MockTextMessageStartEvent(message_id="msg-3", role="assistant"),
        MockTextMessageContentEvent(message_id="msg-3", delta="It's sunny and 70F"),
        MockTextMessageEndEvent(message_id="msg-3"),
        # User thanks
        MockTextMessageStartEvent(message_id="msg-4", role="user"),
        MockTextMessageContentEvent(message_id="msg-4", delta="Thanks!"),
        MockTextMessageEndEvent(message_id="msg-4"),
    ]

    messages = convert_to_ragas_messages(events, metadata=True)

    # Should have: User, AI, Tool, AI, User
    assert len(messages) == 5
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert isinstance(messages[3], AIMessage)
    assert isinstance(messages[4], HumanMessage)

    # Check content
    assert "weather" in messages[0].content.lower()
    assert "check" in messages[1].content.lower()
    assert "sunny" in messages[2].content.lower()
    assert "sunny" in messages[3].content.lower()
    assert "thanks" in messages[4].content.lower()

    # Check metadata
    assert all(msg.metadata is not None for msg in messages)
    assert all("run_id" in msg.metadata for msg in messages)
