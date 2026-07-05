from ragas.integrations.swarm import convert_to_ragas_messages
from ragas.messages import AIMessage


def test_assistant_message_without_tool_calls():
    messages = [{"role": "assistant", "content": "hi"}]

    result = convert_to_ragas_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "hi"
    assert result[0].tool_calls == []
