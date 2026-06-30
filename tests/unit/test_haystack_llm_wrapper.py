from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.prompt_values import StringPromptValue

from ragas.llms.haystack_wrapper import HaystackLLMWrapper, _extract_haystack_reply_text


def _require_haystack():
    pytest.importorskip("haystack")
    from haystack.components.generators.chat.openai import OpenAIChatGenerator
    from haystack.components.generators.openai import OpenAIGenerator
    from haystack.dataclasses import ChatMessage

    return OpenAIChatGenerator, OpenAIGenerator, ChatMessage


@pytest.fixture
def mock_async_pipeline():
    with patch("haystack.AsyncPipeline") as pipeline_cls:
        pipeline = MagicMock()
        pipeline_cls.return_value = pipeline
        yield pipeline


def test_extract_haystack_reply_text_handles_string_and_chat_message():
    _, _, ChatMessage = _require_haystack()

    assert _extract_haystack_reply_text("plain") == "plain"
    assert _extract_haystack_reply_text(ChatMessage.from_assistant("chat")) == "chat"
    assert _extract_haystack_reply_text(None) == ""


def test_haystack_chat_generator_is_accepted(mock_async_pipeline):
    OpenAIChatGenerator, _, _ = _require_haystack()

    generator = OpenAIChatGenerator.__new__(OpenAIChatGenerator)
    generator.run = MagicMock(return_value={"replies": []})

    wrapper = HaystackLLMWrapper(generator)

    assert wrapper._uses_chat_generator is True
    mock_async_pipeline.add_component.assert_called_once_with("llm", generator)


def test_generate_text_with_chat_generator_reply(mock_async_pipeline):
    OpenAIChatGenerator, _, ChatMessage = _require_haystack()

    generator = OpenAIChatGenerator.__new__(OpenAIChatGenerator)
    generator.run = MagicMock(
        return_value={"replies": [ChatMessage.from_assistant("chat reply")]}
    )

    wrapper = HaystackLLMWrapper(generator)
    result = wrapper.generate_text(StringPromptValue(text="hello"))

    generator.run.assert_called_once_with(
        messages="hello",
        generation_kwargs={"temperature": 0.01},
    )
    assert result.generations[0][0].text == "chat reply"


def test_generate_text_with_legacy_generator_reply(mock_async_pipeline):
    _, OpenAIGenerator, _ = _require_haystack()

    generator = OpenAIGenerator.__new__(OpenAIGenerator)
    generator.run = MagicMock(return_value={"replies": ["legacy reply"]})

    wrapper = HaystackLLMWrapper(generator)
    result = wrapper.generate_text(StringPromptValue(text="hello"))

    generator.run.assert_called_once_with(
        prompt="hello",
        generation_kwargs={"temperature": 0.01},
    )
    assert result.generations[0][0].text == "legacy reply"


@pytest.mark.asyncio
async def test_agenerate_text_with_chat_generator_uses_messages_input(
    mock_async_pipeline,
):
    OpenAIChatGenerator, _, ChatMessage = _require_haystack()

    generator = OpenAIChatGenerator.__new__(OpenAIChatGenerator)
    generator.run = MagicMock()
    mock_async_pipeline.run_async = AsyncMock(
        return_value={"llm": {"replies": [ChatMessage.from_assistant("async chat")]}}
    )

    wrapper = HaystackLLMWrapper(generator)
    result = await wrapper.agenerate_text(
        StringPromptValue(text="hello"),
        temperature=0.2,
    )

    mock_async_pipeline.run_async.assert_awaited_once_with(
        data={"llm": {"messages": "hello", "generation_kwargs": {"temperature": 0.2}}}
    )
    assert result.generations[0][0].text == "async chat"


@pytest.mark.asyncio
async def test_agenerate_text_with_legacy_generator_uses_prompt_input(
    mock_async_pipeline,
):
    _, OpenAIGenerator, _ = _require_haystack()

    generator = OpenAIGenerator.__new__(OpenAIGenerator)
    generator.run = MagicMock()
    mock_async_pipeline.run_async = AsyncMock(
        return_value={"llm": {"replies": ["async legacy"]}}
    )

    wrapper = HaystackLLMWrapper(generator)
    result = await wrapper.agenerate_text(
        StringPromptValue(text="hello"),
        temperature=0.2,
    )

    mock_async_pipeline.run_async.assert_awaited_once_with(
        data={"llm": {"prompt": "hello", "generation_kwargs": {"temperature": 0.2}}}
    )
    assert result.generations[0][0].text == "async legacy"
