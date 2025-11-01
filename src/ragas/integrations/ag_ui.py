"""
AG-UI Protocol Integration for Ragas.

This module provides conversion utilities and evaluation functions for AG-UI
protocol agents. It supports converting AG-UI streaming events to Ragas message
format and evaluating AG-UI FastAPI endpoints.

AG-UI is an event-based protocol for agent-to-UI communication that uses typed
events for streaming text messages, tool calls, and state synchronization. This
integration supports both streaming events (Start-Content-End triads) and
convenience chunk events (TextMessageChunk, ToolCallChunk) for complete messages.

Functions:
    convert_to_ragas_messages: Convert AG-UI event sequences to Ragas messages
    convert_messages_snapshot: Convert AG-UI message snapshots to Ragas messages
    evaluate_ag_ui_agent: Batch evaluate an AG-UI FastAPI endpoint

Examples:
    Convert streaming AG-UI events to Ragas messages::

        from ragas.integrations.ag_ui import convert_to_ragas_messages
        from ag_ui.core import Event

        # List of AG-UI events from agent run
        ag_ui_events: List[Event] = [...]

        # Convert to Ragas messages
        ragas_messages = convert_to_ragas_messages(ag_ui_events, metadata=True)

    Evaluate an AG-UI agent endpoint::

        from ragas.integrations.ag_ui import evaluate_ag_ui_agent
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import AspectCritic

        dataset = EvaluationDataset(samples=[
            SingleTurnSample(user_input="What's the weather in SF?")
        ])

        result = await evaluate_ag_ui_agent(
            endpoint_url="http://localhost:8000/agent",
            dataset=dataset,
            metrics=[AspectCritic()]
        )
"""

from __future__ import annotations

import json
import logging
import math
import typing as t
from typing import Any, Dict, List, Optional, Union
import uuid

from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample
from ragas.evaluation import evaluate as ragas_evaluate
from ragas.executor import Executor
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from ragas.metrics.base import Metric

logger = logging.getLogger(__name__)


# Lazy imports for ag_ui to avoid hard dependency
def _import_ag_ui_core():
    """Import AG-UI core types with helpful error message."""
    try:
        from ag_ui.core import (
            BaseEvent,
            Event,
            EventType,
            MessagesSnapshotEvent,
            TextMessageChunkEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageStartEvent,
            ToolCallArgsEvent,
            ToolCallChunkEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            ToolCallStartEvent,
        )

        return (
            BaseEvent,
            Event,
            EventType,
            MessagesSnapshotEvent,
            TextMessageStartEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageChunkEvent,
            ToolCallStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            ToolCallChunkEvent,
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
    incrementally through Start->Content->End event sequences (triads). This
    collector accumulates these events and reconstructs complete Ragas messages.
    It also supports convenience chunk events (TextMessageChunk, ToolCallChunk)
    for complete messages delivered in a single event.

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
        - Text message events: Accumulate and reconstruct messages (streaming triads or chunks)
        - Tool call events: Reconstruct tool calls and results (streaming triads or chunks)
        - Other events: Silently ignored
        """
        (
            BaseEvent,
            Event,
            EventType,
            MessagesSnapshotEvent,
            TextMessageStartEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageChunkEvent,
            ToolCallStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            ToolCallChunkEvent,
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
        elif event_type == EventType.TEXT_MESSAGE_CHUNK:
            self._handle_text_message_chunk(event)

        # Handle tool call events
        elif event_type == EventType.TOOL_CALL_START:
            self._handle_tool_call_start(event)
        elif event_type == EventType.TOOL_CALL_ARGS:
            self._handle_tool_call_args(event)
        elif event_type == EventType.TOOL_CALL_END:
            self._handle_tool_call_end(event)
        elif event_type == EventType.TOOL_CALL_RESULT:
            self._handle_tool_call_result(event)
        elif event_type == EventType.TOOL_CALL_CHUNK:
            self._handle_tool_call_chunk(event)

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

    def _handle_text_message_chunk(self, event: Any) -> None:
        """
        Process a TextMessageChunkEvent - a convenience event combining start, content, and end.

        This handler processes complete messages available at once, bypassing the
        Start-Content-End streaming sequence.
        """
        # Extract message data from chunk event
        message_id = getattr(event, "message_id", None)
        role = getattr(event, "role", "assistant")
        content = getattr(event, "delta", "")

        # Build metadata if requested
        metadata = None
        if self.include_metadata:
            metadata = {
                "timestamp": event.timestamp,
            }
            if message_id:
                metadata["message_id"] = message_id
            if self._current_run_id:
                metadata["run_id"] = self._current_run_id
            if self._current_thread_id:
                metadata["thread_id"] = self._current_thread_id
            if self._current_step:
                metadata["step_name"] = self._current_step

        # Convert to appropriate Ragas message type
        if role == "assistant":
            # Check if there are completed tool calls for this message
            tool_calls = None
            if self._completed_tool_calls:
                tool_calls = list(self._completed_tool_calls.values())
                self._completed_tool_calls.clear()

            self.messages.append(
                AIMessage(content=content, tool_calls=tool_calls, metadata=metadata)
            )
        elif role == "user":
            self.messages.append(HumanMessage(content=content, metadata=metadata))
        else:
            logger.warning(f"Unexpected message role in chunk event: {role}")

    def _handle_tool_call_chunk(self, event: Any) -> None:
        """
        Process a ToolCallChunkEvent - a convenience event combining tool call specification.

        This handler processes complete tool calls available at once, bypassing the
        Start-Args-End streaming sequence.
        """
        # Extract tool call data from chunk event
        tool_call_id = getattr(event, "tool_call_id", None)
        tool_call_name = getattr(event, "tool_call_name", None)
        args_delta = getattr(event, "delta", None)

        if not tool_call_name:
            logger.warning("Received ToolCallChunk without tool_call_name")
            return

        # Parse tool arguments from delta if provided
        args = {}
        if args_delta:
            if isinstance(args_delta, str):
                try:
                    args = json.loads(args_delta)
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse tool call arguments for {tool_call_name}: {args_delta}"
                    )
                    args = {"raw_args": args_delta}
            elif isinstance(args_delta, dict):
                args = args_delta
            else:
                args = {"raw_args": str(args_delta)}

        # Store completed tool call for association with next AI message
        if tool_call_id:
            self._completed_tool_calls[tool_call_id] = ToolCall(
                name=tool_call_name, args=args
            )
        else:
            # If no ID provided, generate one
            temp_id = f"chunk_{len(self._completed_tool_calls)}"
            self._completed_tool_calls[temp_id] = ToolCall(
                name=tool_call_name, args=args
            )

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
        BaseEvent,
        Event,
        EventType,
        MessagesSnapshotEvent,
        TextMessageStartEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageChunkEvent,
        ToolCallStartEvent,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallChunkEvent,
    ) = _import_ag_ui_core()

    if not isinstance(snapshot_event, MessagesSnapshotEvent):
        raise TypeError(
            f"Expected MessagesSnapshotEvent, got {type(snapshot_event).__name__}"
        )

    collector = AGUIEventCollector(metadata=metadata)
    collector._handle_messages_snapshot(snapshot_event)
    return collector.get_messages()


async def _call_ag_ui_endpoint(
    endpoint_url: str,
    user_input: str,
    thread_id: Optional[str] = None,
    agent_config: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> List[Any]:
    """
    Call an AG-UI FastAPI endpoint and collect streaming events.

    Makes an HTTP POST request to an AG-UI compatible FastAPI endpoint
    and parses the Server-Sent Events (SSE) stream to collect all events.

    Parameters
    ----------
    endpoint_url : str
        The URL of the AG-UI FastAPI endpoint (e.g., "http://localhost:8000/agent").
    user_input : str
        The user message/query to send to the agent.
    thread_id : str, optional
        Optional thread ID for conversation continuity.
    agent_config : dict, optional
        Optional agent configuration parameters.
    timeout : float, optional
        Request timeout in seconds (default: 60.0).
    extra_headers : dict, optional
        Optional extra HTTP headers to include in the request (default: None).
        These will be merged with the default "Accept: text/event-stream" header.

    Returns
    -------
    List[Event]
        List of AG-UI events collected from the SSE stream.

    Raises
    ------
    ImportError
        If httpx is not installed.
    httpx.HTTPError
        If the HTTP request fails.

    Notes
    -----
    This function expects the endpoint to return Server-Sent Events (SSE)
    with content type "text/event-stream". Each event should be in the format:

        data: {"type": "...", ...}\\n\\n

    The function will parse the SSE stream and deserialize each event
    using AG-UI's RunAgentInput model.
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError(
            "AG-UI FastAPI integration requires httpx. "
            "Install it with: pip install httpx"
        ) from e

    # Import AG-UI types
    try:
        from ag_ui.core import Event, RunAgentInput, UserMessage
        from pydantic import TypeAdapter
    except ImportError as e:
        raise ImportError(
            "AG-UI integration requires the ag-ui-protocol package. "
            "Install it with: pip install ag-ui-protocol"
        ) from e

    # Create TypeAdapter for Event discriminated union
    # This properly handles the union of all event types based on the 'type' discriminator
    event_adapter = TypeAdapter(Event)

    # Prepare request payload
    payload = RunAgentInput(
        thread_id=thread_id or f"thread_{uuid.uuid4()}",  # Generate thread ID if not provided
        run_id=f"run_{uuid.uuid4()}",  # Generate a unique run ID
        messages=[
            UserMessage(id="1", content=user_input)
        ],
        state={},
        tools=[],
        context=[],
        forwarded_props={}
    )

    # Collect events from SSE stream
    events: List[Any] = []

    # Merge default headers with extra headers
    headers = {"Accept": "text/event-stream"}
    if extra_headers:
        headers.update(extra_headers)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream(
            "POST",
            endpoint_url,
            json=payload.model_dump(exclude_none=True),
            headers=headers,
        ) as response:
            response.raise_for_status()

            # Parse SSE stream line by line
            async for line in response.aiter_lines():
                line = line.strip()

                # SSE format: "data: {...}"
                if line.startswith("data: "):
                    json_data = line[6:]  # Remove "data: " prefix

                    try:
                        # Parse JSON and convert to Event using TypeAdapter
                        # TypeAdapter properly handles discriminated unions based on 'type' field
                        event_dict = json.loads(json_data)
                        event = event_adapter.validate_python(event_dict)
                        events.append(event)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse SSE event: {e}")
                        continue

    return events


async def evaluate_ag_ui_agent(
    endpoint_url: str,
    dataset: EvaluationDataset,
    metrics: List["Metric"],
    metadata: bool = False,
    run_config: Optional[RunConfig] = None,
    batch_size: Optional[int] = None,
    raise_exceptions: bool = False,
    show_progress: bool = True,
    timeout: float = 60.0,
    evaluator_llm: Optional[Any] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> EvaluationResult:
    """
    Evaluate an AG-UI agent by calling its FastAPI endpoint with test queries.

    This function runs a batch evaluation by:
    1. Calling the AG-UI FastAPI endpoint for each query in the dataset
    2. Collecting streaming AG-UI events from each response
    3. Converting events to Ragas message format
    4. Evaluating with specified metrics

    Parameters
    ----------
    endpoint_url : str
        URL of the AG-UI FastAPI endpoint (e.g., "http://localhost:8000/agent").
    dataset : EvaluationDataset
        Dataset containing test queries (user_input field).
    metrics : List[Metric]
        List of Ragas metrics to evaluate (e.g., AspectCritic, Faithfulness).
    metadata : bool, optional
        Whether to include AG-UI metadata in converted messages (default: False).
    run_config : RunConfig, optional
        Configuration for the evaluation run.
    batch_size : int, optional
        Number of queries to process in parallel (default: None = auto).
    raise_exceptions : bool, optional
        Whether to raise exceptions or log warnings (default: False).
    show_progress : bool, optional
        Whether to show progress bar (default: True).
    timeout : float, optional
        HTTP request timeout in seconds (default: 60.0).
    evaluator_llm : Any, optional
        Optional LLM to use for evaluation metrics (default: None).
    extra_headers : dict, optional
        Optional extra HTTP headers to include in requests to the agent endpoint (default: None).
        These will be merged with the default "Accept: text/event-stream" header.

    Returns
    -------
    EvaluationResult
        Results containing metric scores for the dataset.

    Raises
    ------
    ImportError
        If required packages (httpx, ag-ui-protocol) are not installed.
    ValueError
        If dataset is not of type EvaluationDataset or is multi-turn.

    Examples
    --------
    Evaluate an AG-UI agent endpoint with standard metrics::

        >>> from ragas.integrations.ag_ui import evaluate_ag_ui_agent
        >>> from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        >>> from ragas.metrics import AspectCritic, Faithfulness
        >>>
        >>> dataset = EvaluationDataset(samples=[
        ...     SingleTurnSample(
        ...         user_input="What's the weather in San Francisco?",
        ...         reference="Use the weather API to check SF weather"
        ...     )
        ... ])
        >>>
        >>> result = await evaluate_ag_ui_agent(
        ...     endpoint_url="http://localhost:8000/agent",
        ...     dataset=dataset,
        ...     metrics=[AspectCritic(), Faithfulness()]
        ... )

    With AG-UI metadata included::

        >>> result = await evaluate_ag_ui_agent(
        ...     endpoint_url="http://localhost:8000/agent",
        ...     dataset=dataset,
        ...     metrics=[AspectCritic()],
        ...     metadata=True  # Include run_id, thread_id, etc.
        ... )

    Notes
    -----
    - The endpoint must return Server-Sent Events (SSE) with AG-UI protocol events
    - Each query is sent as a separate HTTP request with RunAgentInput payload
    - Queries are executed in parallel using Ragas Executor
    - Failed queries are logged and recorded as NaN in results
    - Multi-turn conversations are not yet supported

    See Also
    --------
    convert_to_ragas_messages : Convert AG-UI events to Ragas messages
    _call_ag_ui_endpoint : HTTP client helper for calling endpoints
    """
    # Validate dataset
    if dataset is None or not isinstance(dataset, EvaluationDataset):
        raise ValueError("Please provide a dataset that is of type EvaluationDataset")

    # Check if multi-turn
    if dataset.is_multi_turn():
        raise NotImplementedError(
            "Multi-turn evaluation for AG-UI agents is not implemented yet. "
            "Please raise an issue on GitHub if you need this feature."
        )

    samples = t.cast(List[SingleTurnSample], dataset.samples)

    # Create executor for parallel HTTP calls
    executor = Executor(
        desc="Calling AG-UI Agent",
        keep_progress_bar=True,
        show_progress=show_progress,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
        batch_size=batch_size,
    )

    # Submit HTTP calls for all queries
    queries = [sample.user_input for sample in samples]
    for i, query in enumerate(queries):
        executor.submit(
            _call_ag_ui_endpoint,
            endpoint_url=endpoint_url,
            user_input=query,
            thread_id=f"thread-eval-{i}",
            agent_config=None,
            timeout=timeout,
            extra_headers=extra_headers,
        )

    # Collect results and convert to messages
    responses: List[Optional[str]] = []
    retrieved_contexts: List[Optional[List[str]]] = []
    results = executor.results()

    for i, result in enumerate(results):
        # Handle failed jobs which are recorded as NaN in the executor
        if isinstance(result, float) and math.isnan(result):
            responses.append(None)
            retrieved_contexts.append(None)
            logger.warning(
                f"AG-UI agent call failed for query {i}: '{queries[i]}'"
            )
            continue

        # Convert AG-UI events to Ragas messages
        events = t.cast(List[Any], result)
        try:
            logger.info(f"Processing query {i}, received {len(events)} events")
            messages = convert_to_ragas_messages(events, metadata=metadata)
            logger.info(f"Converted to {len(messages)} messages")

            # Extract response text from AI messages
            response_text = ""
            context_list: List[str] = []

            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content:
                    response_text += msg.content
                    logger.debug(f"Found AI message with content: {msg.content[:100]}...")
                # Tool results could contain retrieved context
                elif isinstance(msg, ToolMessage) and msg.content:
                    context_list.append(msg.content)
                    logger.debug(f"Found tool message with content: {msg.content[:100]}...")

            logger.info(f"Query {i} - Response length: {len(response_text)}, Contexts: {len(context_list)}")
            responses.append(response_text or None)
            retrieved_contexts.append(context_list if context_list else None)

        except Exception as e:
            logger.warning(
                f"Failed to convert events for query {i}: {e}", exc_info=True
            )
            responses.append(None)
            retrieved_contexts.append(None)

    # Update samples in place with responses and retrieved_contexts
    # This ensures the dataset includes all fields needed for evaluation
    for i, sample in enumerate(samples):
        sample.response = responses[i] if responses[i] is not None else ""
        sample.retrieved_contexts = retrieved_contexts[i] if retrieved_contexts[i] is not None else []

    # Run evaluation with metrics
    evaluation_result = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=raise_exceptions,
        show_progress=show_progress,
        run_config=run_config or RunConfig(),
        return_executor=False,
        llm=evaluator_llm,
    )

    # Type assertion since return_executor=False guarantees EvaluationResult
    return t.cast(EvaluationResult, evaluation_result)
