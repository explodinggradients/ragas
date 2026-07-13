from ragas.integrations.swarm import convert_to_ragas_messages
from ragas.messages import AIMessage, HumanMessage, ToolMessage


def test_convert_swarm_messages_no_tool_calls():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    converted = convert_to_ragas_messages(messages)
    assert len(converted) == 2
    assert isinstance(converted[0], HumanMessage)
    assert converted[0].content == "hello"
    assert isinstance(converted[1], AIMessage)
    assert converted[1].content == "hi there"
    assert converted[1].tool_calls == []


def test_convert_swarm_messages_with_tool_calls():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "Sunny, 20C"},
    ]
    converted = convert_to_ragas_messages(messages)
    assert len(converted) == 2
    assert isinstance(converted[0], AIMessage)
    assert len(converted[0].tool_calls) == 1
    assert converted[0].tool_calls[0].name == "get_weather"
    assert converted[0].tool_calls[0].args == {"location": "San Francisco"}
    assert isinstance(converted[1], ToolMessage)
    assert converted[1].content == "Sunny, 20C"
