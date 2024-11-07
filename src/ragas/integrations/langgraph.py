import json
from typing import List, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import ragas.messages as r


def convert_to_ragas_messages(
    messages: List[Union[HumanMessage, SystemMessage, AIMessage, ToolMessage]]
) -> List[Union[r.HumanMessage, r.AIMessage, r.ToolMessage]]:
    """
    Converts LangChain messages into Ragas messages for agent evaluation.

    Args:
        messages: List of LangChain message objects (HumanMessage, SystemMessage,
                 AIMessage, ToolMessage)

    Returns:
        List of corresponding Ragas message objects

    Raises:
        ValueError: If an unsupported message type is encountered
    """
    MESSAGE_TYPE_MAP = {
        HumanMessage: lambda m: r.HumanMessage(content=m.content),
        ToolMessage: lambda m: r.ToolMessage(content=m.content),
    }

    def _extract_tool_calls(message: AIMessage) -> List[r.ToolCall]:
        tool_calls = message.additional_kwargs.get("tool_calls", [])
        return [
            r.ToolCall(
                name=tool_call["function"]["name"],
                args=json.loads(tool_call["function"]["arguments"]),
            )
            for tool_call in tool_calls
        ]

    def _convert_ai_message(message: AIMessage) -> r.AIMessage:
        tool_calls = _extract_tool_calls(message) if message.additional_kwargs else None
        return r.AIMessage(content=message.content, tool_calls=tool_calls)

    def _convert_message(message):
        if isinstance(message, SystemMessage):
            return None  # Skip SystemMessages
        if isinstance(message, AIMessage):
            return _convert_ai_message(message)
        converter = MESSAGE_TYPE_MAP.get(type(message))
        if converter is None:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")
        return converter(message)

    return [
        converted
        for message in messages
        if (converted := _convert_message(message)) is not None
    ]
