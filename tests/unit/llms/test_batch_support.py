"""
Unit tests for batch API support in LLM wrappers.
"""

from typing import List, cast
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import PromptValue
from langchain_openai import ChatOpenAI

from ragas.llms.base import LangchainLLMWrapper
from ragas.llms.batch_api import BatchRequest


class MockPromptValue(PromptValue):
    """Mock implementation of PromptValue for testing."""

    def __init__(self, messages: List[BaseMessage]):
        self._messages = messages

    def to_messages(self) -> List[BaseMessage]:
        return self._messages

    def to_string(self) -> str:
        return " ".join([str(msg.content) for msg in self._messages])


class TestLangchainLLMWrapperBatchSupport:
    """Test batch API support in LangchainLLMWrapper."""

    def test_supports_batch_api_openai(self):
        """Test batch API support detection for OpenAI models."""
        mock_openai_llm = Mock(spec=ChatOpenAI)
        wrapper = LangchainLLMWrapper(mock_openai_llm)

        assert wrapper.supports_batch_api()
        assert wrapper.batch_api_support

    def test_supports_batch_api_non_openai(self):
        """Test batch API support detection for non-OpenAI models."""
        mock_other_llm = Mock()
        mock_other_llm.__class__.__name__ = "ChatAnthropic"
        wrapper = LangchainLLMWrapper(mock_other_llm)

        assert not wrapper.supports_batch_api()
        assert not wrapper.batch_api_support

    @patch("ragas.llms.batch_api.create_batch_api")
    def test_get_batch_api(self, mock_create_batch_api):
        """Test getting batch API instance."""
        mock_openai_llm = Mock(spec=ChatOpenAI)
        mock_openai_llm.client = Mock()

        mock_batch_api = Mock()
        mock_create_batch_api.return_value = mock_batch_api

        wrapper = LangchainLLMWrapper(mock_openai_llm)
        batch_api = wrapper._get_batch_api()

        assert batch_api == mock_batch_api
        mock_create_batch_api.assert_called_once_with(mock_openai_llm.client)

    def test_get_batch_api_not_supported(self):
        """Test getting batch API when not supported."""
        mock_other_llm = Mock()
        wrapper = LangchainLLMWrapper(mock_other_llm)

        with pytest.raises(ValueError, match="Batch API not supported"):
            wrapper._get_batch_api()

    def test_create_batch_requests_from_prompts(self):
        """Test converting prompts to batch requests."""
        mock_openai_llm = Mock(spec=ChatOpenAI)
        mock_openai_llm.model_name = "gpt-4o-mini"

        wrapper = LangchainLLMWrapper(mock_openai_llm)

        # Create proper prompt objects
        prompt1 = MockPromptValue([HumanMessage(content="Hello, how are you?")])

        prompt2 = MockPromptValue(
            [
                SystemMessage(content="You are helpful"),
                HumanMessage(content="What is AI?"),
            ]
        )

        prompts = cast(List[PromptValue], [prompt1, prompt2])

        requests = wrapper._create_batch_requests_from_prompts(
            prompts=prompts, n=1, temperature=0.7, stop=["END"]
        )

        assert len(requests) == 2
        assert all(isinstance(req, BatchRequest) for req in requests)

        # Check first request
        assert requests[0].custom_id == "ragas-batch-0"
        assert requests[0].body["model"] == "gpt-4o-mini"
        assert requests[0].body["temperature"] == 0.7
        assert requests[0].body["stop"] == ["END"]
        assert len(requests[0].body["messages"]) == 1
        assert requests[0].body["messages"][0]["role"] == "human"

        # Check second request
        assert requests[1].custom_id == "ragas-batch-1"
        assert len(requests[1].body["messages"]) == 2
        assert requests[1].body["messages"][0]["role"] == "system"
        assert requests[1].body["messages"][1]["role"] == "human"

    def test_create_batch_requests_string_fallback(self):
        """Test batch request creation with string prompt fallback."""
        mock_openai_llm = Mock(spec=ChatOpenAI)
        mock_openai_llm.model_name = "gpt-3.5-turbo"

        wrapper = LangchainLLMWrapper(mock_openai_llm)

        # Create prompt without to_messages method to trigger fallback
        # Use a class that inherits from object and doesn't have to_messages
        class StringOnlyPrompt:
            def __str__(self):
                return "What is Python?"

        prompt = StringOnlyPrompt()

        requests = wrapper._create_batch_requests_from_prompts(
            cast(List[PromptValue], [prompt])
        )

        assert len(requests) == 1
        assert requests[0].body["messages"][0]["role"] == "user"
        assert requests[0].body["messages"][0]["content"] == "What is Python?"

    def test_create_batch_requests_bypass_temperature(self):
        """Test batch request creation with temperature bypass."""
        mock_openai_llm = Mock(spec=ChatOpenAI)
        mock_openai_llm.model_name = "gpt-4o"

        wrapper = LangchainLLMWrapper(mock_openai_llm, bypass_temperature=True)

        prompt = MockPromptValue([HumanMessage(content="Hello")])

        requests = wrapper._create_batch_requests_from_prompts(
            cast(List[PromptValue], [prompt]), temperature=0.5
        )

        # Temperature should be removed due to bypass_temperature=True
        assert "temperature" not in requests[0].body

    @patch.object(LangchainLLMWrapper, "_get_batch_api")
    @patch.object(LangchainLLMWrapper, "_create_batch_requests_from_prompts")
    def test_create_batch_job(self, mock_create_requests, mock_get_batch_api):
        """Test creating batch job."""
        mock_openai_llm = Mock(spec=ChatOpenAI)
        wrapper = LangchainLLMWrapper(mock_openai_llm)

        mock_batch_api = Mock()
        mock_batch_job = Mock()
        mock_batch_api.create_batch.return_value = mock_batch_job
        mock_get_batch_api.return_value = mock_batch_api

        mock_requests = [Mock(), Mock()]
        mock_create_requests.return_value = mock_requests

        prompts = cast(
            List[PromptValue],
            [
                MockPromptValue([HumanMessage(content="Test prompt 1")]),
                MockPromptValue([HumanMessage(content="Test prompt 2")]),
            ],
        )
        metadata = {"test": "value"}

        result = wrapper.create_batch_job(
            prompts=prompts, n=2, temperature=0.8, stop=["STOP"], metadata=metadata
        )

        assert result == mock_batch_job
        mock_get_batch_api.assert_called_once()
        mock_create_requests.assert_called_once_with(prompts, 2, 0.8, ["STOP"])
        mock_batch_api.create_batch.assert_called_once_with(
            requests=mock_requests, metadata=metadata
        )

    def test_create_batch_job_not_supported(self):
        """Test creating batch job when not supported."""
        mock_other_llm = Mock()
        wrapper = LangchainLLMWrapper(mock_other_llm)

        with pytest.raises(ValueError, match="Batch API not supported"):
            wrapper.create_batch_job(
                cast(
                    List[PromptValue], [MockPromptValue([HumanMessage(content="test")])]
                )
            )
