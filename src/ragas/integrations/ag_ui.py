"""
AG-UI Protocol Integration for Ragas.

This module provides conversion utilities to transform AG-UI protocol events into
Ragas message format for evaluation. It supports both streaming event sequences
and complete message snapshots.

AG-UI is an event-based protocol for agent-to-UI communication that uses typed
events for streaming text messages, tool calls, and state synchronization.

Example:
    Convert streaming AG-UI events to Ragas messages::

        from ragas.integrations.ag_ui import convert_to_ragas_messages
        from ag_ui.core import Event

        # List of AG-UI events from agent run
        ag_ui_events: List[Event] = [...]

        # Convert to Ragas messages
        ragas_messages = convert_to_ragas_messages(ag_ui_events, metadata=True)

    Convert a messages snapshot::

        from ragas.integrations.ag_ui import convert_messages_snapshot
        from ag_ui.core import MessagesSnapshotEvent

        snapshot = MessagesSnapshotEvent(messages=[...])
        ragas_messages = convert_messages_snapshot(snapshot)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

logger = logging.getLogger(__name__)


# Lazy imports for ag_ui to avoid hard dependency
def _import_ag_ui_core():
    """Import AG-UI core types with helpful error message."""
    try:
        from ag_ui.core import (
            Event,
            EventType,
            MessagesSnapshotEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            ToolCallStartEvent,
        )

        return (
            Event,
            EventType,
            MessagesSnapshotEvent,
            TextMessageStartEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            ToolCallStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
        )
    except ImportError as e:
        raise ImportError(
            "AG-UI integration requires the ag-ui-protocol package. "
            "Install it with: pip install ag-ui-protocol"
        ) from e


class AGUIEventCollector:
    """
    Collects and reconstructs complete messages from streaming AG-UI events.

    AG-UI uses an event-based streaming protocol where messages are delivered
    incrementally through Start->Content->End event sequences. This collector
    accumulates these events and reconstructs complete Ragas messages.

    Attributes
    ----------
    messages : List[Union[HumanMessage, AIMessage, ToolMessage]]
        Accumulated complete messages ready for Ragas evaluation.
    include_metadata : bool
        Whether to include AG-UI metadata in converted messages.

    Example
    -------
    >>> collector = AGUIEventCollector(metadata=True)
    >>> for event in ag_ui_event_stream:
    ...     collector.process_event(event)
    >>> ragas_messages = collector.get_messages()
    """

    def __init__(self, metadata: bool = False):
        """
        Initialize the event collector.

        Parameters
        ----------
        metadata : bool, optional
            Whether to include AG-UI event metadata in Ragas messages (default: False)
        """
        self.include_metadata = metadata
        self.messages: List[Union[HumanMessage, AIMessage, ToolMessage]] = []

        # State tracking for streaming message reconstruction
        self._active_text_messages: Dict[str, Dict[str, Any]] = {}
        self._active_tool_calls: Dict[str, Dict[str, Any]] = {}
        self._completed_tool_calls: Dict[str, ToolCall] = {}

        # Context tracking for metadata
        self._current_run_id: Optional[str] = None
        self._current_thread_id: Optional[str] = None
        self._current_step: Optional[str] = None

    def process_event(self, event: Any) -> None:
        """
        Process a single AG-UI event and update internal state.

        Parameters
        ----------
        event : Event
            An AG-UI protocol event from ag_ui.core

        Notes
        -----
        This method handles different event types:
        - Lifecycle events (RUN_STARTED, STEP_STARTED): Update context
        - Text message events: Accumulate and reconstruct messages
        - Tool call events: Reconstruct tool calls and results
        - Other events: Silently ignored
        """
        (
            Event,
            EventType,
            MessagesSnapshotEvent,
            TextMessageStartEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            ToolCallStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
        ) = _import_ag_ui_core()

        event_type = event.type

        # Update context from lifecycle events
        if event_type == EventType.RUN_STARTED:
            self._current_run_id = event.run_id
            self._current_thread_id = event.thread_id
        elif event_type == EventType.STEP_STARTED:
            self._current_step = event.step_name
        elif event_type == EventType.STEP_FINISHED:
            if event.step_name == self._current_step:
                self._current_step = None

        # Handle text message events
        elif event_type == EventType.TEXT_MESSAGE_START:
            self._handle_text_message_start(event)
        elif event_type == EventType.TEXT_MESSAGE_CONTENT:
            self._handle_text_message_content(event)
        elif event_type == EventType.TEXT_MESSAGE_END:
            self._handle_text_message_end(event)

        # Handle tool call events
        elif event_type == EventType.TOOL_CALL_START:
            self._handle_tool_call_start(event)
        elif event_type == EventType.TOOL_CALL_ARGS:
            self._handle_tool_call_args(event)
        elif event_type == EventType.TOOL_CALL_END:
            self._handle_tool_call_end(event)
        elif event_type == EventType.TOOL_CALL_RESULT:
            self._handle_tool_call_result(event)

        # MessagesSnapshot provides complete history
        elif event_type == EventType.MESSAGES_SNAPSHOT:
            self._handle_messages_snapshot(event)

        # Ignore lifecycle, state management, and other events
        else:
            logger.debug(f"Ignoring AG-UI event type: {event_type}")

    def _handle_text_message_start(self, event: Any) -> None:
        """Initialize a new streaming text message."""
        self._active_text_messages[event.message_id] = {
            "message_id": event.message_id,
            "role": event.role,
            "content_chunks": [],
            "timestamp": event.timestamp,
        }

    def _handle_text_message_content(self, event: Any) -> None:
        """Accumulate text content chunk for a streaming message."""
        if event.message_id in self._active_text_messages:
            self._active_text_messages[event.message_id]["content_chunks"].append(
                event.delta
            )
        else:
            logger.warning(
                f"Received TextMessageContent for unknown message_id: {event.message_id}"
            )

    def _handle_text_message_end(self, event: Any) -> None:
        """Finalize a streaming text message and convert to Ragas format."""
        if event.message_id not in self._active_text_messages:
            logger.warning(
                f"Received TextMessageEnd for unknown message_id: {event.message_id}"
            )
            return

        msg_data = self._active_text_messages.pop(event.message_id)
        content = "".join(msg_data["content_chunks"])
        role = msg_data["role"]

        # Build metadata if requested
        metadata = None
        if self.include_metadata:
            metadata = {
                "message_id": msg_data["message_id"],
                "timestamp": msg_data["timestamp"],
            }
            if self._current_run_id:
                metadata["run_id"] = self._current_run_id
            if self._current_thread_id:
                metadata["thread_id"] = self._current_thread_id
            if self._current_step:
                metadata["step_name"] = self._current_step

        # Convert to appropriate Ragas message type
        if role == "assistant":
            # Check if there are completed tool calls for this message
            # Tool calls are associated by being emitted before the message end
            tool_calls = None
            if self._completed_tool_calls:
                # Tool calls are accumulated before message ends
                tool_calls = list(self._completed_tool_calls.values())
                self._completed_tool_calls.clear()

            self.messages.append(
                AIMessage(content=content, tool_calls=tool_calls, metadata=metadata)
            )
        elif role == "user":
            self.messages.append(HumanMessage(content=content, metadata=metadata))
        else:
            logger.warning(f"Unexpected message role: {role}")

    def _handle_tool_call_start(self, event: Any) -> None:
        """Initialize a new streaming tool call."""
        self._active_tool_calls[event.tool_call_id] = {
            "tool_call_id": event.tool_call_id,
            "tool_call_name": event.tool_call_name,
            "parent_message_id": getattr(event, "parent_message_id", None),
            "args_chunks": [],
            "timestamp": event.timestamp,
        }

    def _handle_tool_call_args(self, event: Any) -> None:
        """Accumulate tool argument chunks."""
        if event.tool_call_id in self._active_tool_calls:
            self._active_tool_calls[event.tool_call_id]["args_chunks"].append(
                event.delta
            )
        else:
            logger.warning(
                f"Received ToolCallArgs for unknown tool_call_id: {event.tool_call_id}"
            )

    def _handle_tool_call_end(self, event: Any) -> None:
        """Finalize a tool call specification (args are complete, but not yet executed)."""
        if event.tool_call_id not in self._active_tool_calls:
            logger.warning(
                f"Received ToolCallEnd for unknown tool_call_id: {event.tool_call_id}"
            )
            return

        tool_data = self._active_tool_calls.pop(event.tool_call_id)
        args_json = "".join(tool_data["args_chunks"])

        # Parse tool arguments
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse tool call arguments for {tool_data['tool_call_name']}: {args_json}"
            )
            args = {"raw_args": args_json}

        # Store completed tool call for association with next AI message
        self._completed_tool_calls[event.tool_call_id] = ToolCall(
            name=tool_data["tool_call_name"], args=args
        )

    def _handle_tool_call_result(self, event: Any) -> None:
        """Convert tool call result to Ragas ToolMessage."""
        metadata = None
        if self.include_metadata:
            metadata = {
                "tool_call_id": event.tool_call_id,
                "message_id": event.message_id,
                "timestamp": event.timestamp,
            }
            if self._current_run_id:
                metadata["run_id"] = self._current_run_id
            if self._current_thread_id:
                metadata["thread_id"] = self._current_thread_id

        self.messages.append(ToolMessage(content=event.content, metadata=metadata))

    def _handle_messages_snapshot(self, event: Any) -> None:
        """
        Process a MessagesSnapshotEvent containing complete message history.

        This bypasses streaming reconstruction and directly converts
        AG-UI Message objects to Ragas format.
        """
        for msg in event.messages:
            # AG-UI Message structure varies, but typically has role and content
            role = getattr(msg, "role", None)
            content = str(getattr(msg, "content", ""))

            metadata = None
            if self.include_metadata:
                metadata = {"source": "messages_snapshot"}
                if hasattr(msg, "id"):
                    metadata["message_id"] = msg.id

            if role == "assistant":
                # Check for tool calls in message
                tool_calls = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls = [
                        ToolCall(name=tc.name, args=tc.args) for tc in msg.tool_calls
                    ]
                self.messages.append(
                    AIMessage(content=content, tool_calls=tool_calls, metadata=metadata)
                )
            elif role == "user":
                self.messages.append(HumanMessage(content=content, metadata=metadata))
            elif role == "tool":
                self.messages.append(ToolMessage(content=content, metadata=metadata))
            else:
                logger.debug(f"Skipping message with role: {role}")

    def get_messages(self) -> List[Union[HumanMessage, AIMessage, ToolMessage]]:
        """
        Retrieve all accumulated Ragas messages.

        Returns
        -------
        List[Union[HumanMessage, AIMessage, ToolMessage]]
            Complete list of Ragas messages reconstructed from AG-UI events.

        Notes
        -----
        This returns a copy of the accumulated messages. The collector's
        internal state is not cleared, so calling this multiple times
        returns the same messages.
        """
        return self.messages.copy()

    def clear(self) -> None:
        """
        Clear all accumulated messages and reset internal state.

        Useful for reusing the same collector instance for multiple
        conversation sessions.
        """
        self.messages.clear()
        self._active_text_messages.clear()
        self._active_tool_calls.clear()
        self._completed_tool_calls.clear()
        self._current_run_id = None
        self._current_thread_id = None
        self._current_step = None


def convert_to_ragas_messages(
    events: List[Any],
    metadata: bool = False,
) -> List[Union[HumanMessage, AIMessage, ToolMessage]]:
    """
    Convert a sequence of AG-UI protocol events to Ragas message format.

    This function processes AG-UI events and reconstructs complete messages
    from streaming event sequences (Start->Content->End patterns). It handles
    text messages, tool calls, and filters out non-message events like
    lifecycle and state management events.

    Parameters
    ----------
    events : List[Event]
        List of AG-UI protocol events from ag_ui.core. Can contain any mix
        of event types - non-message events are automatically filtered out.
    metadata : bool, optional
        Whether to include AG-UI event metadata (run_id, thread_id, timestamps)
        in the converted Ragas messages (default: False).

    Returns
    -------
    List[Union[HumanMessage, AIMessage, ToolMessage]]
        List of Ragas messages ready for evaluation. Messages preserve
        conversation order and tool call associations.

    Raises
    ------
    ImportError
        If the ag-ui-protocol package is not installed.

    Examples
    --------
    Convert AG-UI events from an agent run::

        >>> from ragas.integrations.ag_ui import convert_to_ragas_messages
        >>> from ag_ui.core import (
        ...     RunStartedEvent, TextMessageStartEvent,
        ...     TextMessageContentEvent, TextMessageEndEvent
        ... )
        >>>
        >>> events = [
        ...     RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        ...     TextMessageStartEvent(message_id="msg-1", role="assistant"),
        ...     TextMessageContentEvent(message_id="msg-1", delta="Hello"),
        ...     TextMessageContentEvent(message_id="msg-1", delta=" world"),
        ...     TextMessageEndEvent(message_id="msg-1"),
        ... ]
        >>> messages = convert_to_ragas_messages(events, metadata=True)
        >>> messages[0].content
        'Hello world'

    Process events with tool calls::

        >>> events = [
        ...     TextMessageStartEvent(message_id="msg-1", role="assistant"),
        ...     TextMessageContentEvent(message_id="msg-1", delta="Let me check"),
        ...     TextMessageEndEvent(message_id="msg-1"),
        ...     ToolCallStartEvent(
        ...         tool_call_id="tc-1",
        ...         tool_call_name="get_weather",
        ...         parent_message_id="msg-1"
        ...     ),
        ...     ToolCallArgsEvent(tool_call_id="tc-1", delta='{"city": "SF"}'),
        ...     ToolCallEndEvent(tool_call_id="tc-1"),
        ...     ToolCallResultEvent(
        ...         tool_call_id="tc-1",
        ...         message_id="result-1",
        ...         content="Sunny, 72°F"
        ...     ),
        ... ]
        >>> messages = convert_to_ragas_messages(events)
        >>> len(messages)
        2  # AI message + Tool result message

    Notes
    -----
    - Streaming events (Start->Content->End) are automatically reconstructed
    - Tool calls are associated with the preceding AI message
    - Non-message events (lifecycle, state) are silently filtered
    - Incomplete event sequences are logged as warnings
    - AG-UI metadata can be preserved in message.metadata when metadata=True

    See Also
    --------
    convert_messages_snapshot : Convert complete message history from snapshot
    AGUIEventCollector : Lower-level API for streaming event collection
    """
    collector = AGUIEventCollector(metadata=metadata)

    for event in events:
        collector.process_event(event)

    return collector.get_messages()


def convert_messages_snapshot(
    snapshot_event: Any,
    metadata: bool = False,
) -> List[Union[HumanMessage, AIMessage, ToolMessage]]:
    """
    Convert an AG-UI MessagesSnapshotEvent to Ragas message format.

    MessagesSnapshotEvent provides a complete conversation history in a
    single event, bypassing the need to reconstruct from streaming events.
    This is more efficient when the complete history is already available.

    Parameters
    ----------
    snapshot_event : MessagesSnapshotEvent
        AG-UI event containing complete message history array.
    metadata : bool, optional
        Whether to include metadata in converted messages (default: False).

    Returns
    -------
    List[Union[HumanMessage, AIMessage, ToolMessage]]
        List of Ragas messages from the snapshot.

    Raises
    ------
    ImportError
        If the ag-ui-protocol package is not installed.

    Examples
    --------
    >>> from ragas.integrations.ag_ui import convert_messages_snapshot
    >>> from ag_ui.core import MessagesSnapshotEvent
    >>>
    >>> snapshot = MessagesSnapshotEvent(messages=[
    ...     {"role": "user", "content": "What's the weather?"},
    ...     {"role": "assistant", "content": "Let me check for you."},
    ... ])
    >>> messages = convert_messages_snapshot(snapshot)
    >>> len(messages)
    2

    Notes
    -----
    This is the preferred method when working with complete conversation
    history. It's faster than processing streaming events and avoids the
    complexity of event sequence reconstruction.

    See Also
    --------
    convert_to_ragas_messages : Convert streaming event sequences
    """
    (
        Event,
        EventType,
        MessagesSnapshotEvent,
        TextMessageStartEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        ToolCallStartEvent,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
    ) = _import_ag_ui_core()

    if not isinstance(snapshot_event, MessagesSnapshotEvent):
        raise TypeError(
            f"Expected MessagesSnapshotEvent, got {type(snapshot_event).__name__}"
        )

    collector = AGUIEventCollector(metadata=metadata)
    collector._handle_messages_snapshot(snapshot_event)
    return collector.get_messages()
