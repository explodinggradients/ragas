import pytest
from unittest.mock import Mock
from pydantic import BaseModel

from ragas_experimental.llm.llm import RagasLLM, ragas_llm


class LLMResponseModel(BaseModel):
    response: str


class MockClient:
    """Mock client that simulates an LLM client."""
    
    def __init__(self, is_async=False):
        self.is_async = is_async
        self.chat = Mock()
        self.chat.completions = Mock()
        if is_async:
            async def async_create(*args, **kwargs):
                return LLMResponseModel(response="Mock response")
            self.chat.completions.create = async_create
        else:
            def sync_create(*args, **kwargs):
                return LLMResponseModel(response="Mock response")
            self.chat.completions.create = sync_create


class MockInstructor:
    """Mock instructor client that wraps the base client."""
    
    def __init__(self, client):
        self.client = client
        self.chat = Mock()
        self.chat.completions = Mock()
        
        if client.is_async:
            # Async client - create a proper async function
            async def async_create(*args, **kwargs):
                return LLMResponseModel(response="Instructor response")
            self.chat.completions.create = async_create
        else:
            # Sync client - create a regular function
            def sync_create(*args, **kwargs):
                return LLMResponseModel(response="Instructor response")
            self.chat.completions.create = sync_create


@pytest.fixture
def mock_sync_client():
    """Create a mock synchronous client."""
    return MockClient(is_async=False)


@pytest.fixture
def mock_async_client():
    """Create a mock asynchronous client.""" 
    return MockClient(is_async=True)


def test_ragas_llm_initialization(mock_sync_client, monkeypatch):
    """Test RagasLLM initialization with different providers."""
    # Mock instructor to return our mock instructor
    def mock_from_openai(client):
        return MockInstructor(client)
    
    monkeypatch.setattr('instructor.from_openai', mock_from_openai)
    
    llm = RagasLLM(
        provider="openai",
        model="gpt-4",
        client=mock_sync_client
    )
    
    assert llm.provider == "openai"
    assert llm.model == "gpt-4"
    assert llm.client is not None
    assert not llm.is_async


def test_ragas_llm_async_detection(mock_async_client, monkeypatch):
    """Test that RagasLLM correctly detects async clients."""
    # Mock instructor to return our mock instructor  
    def mock_from_openai(client):
        return MockInstructor(client)
    
    monkeypatch.setattr('instructor.from_openai', mock_from_openai)
    
    llm = RagasLLM(
        provider="openai",
        model="gpt-4",
        client=mock_async_client
    )
    
    assert llm.is_async


def test_ragas_llm_factory_function(mock_sync_client, monkeypatch):
    """Test the ragas_llm factory function."""
    def mock_from_openai(client):
        return MockInstructor(client)
    
    monkeypatch.setattr('instructor.from_openai', mock_from_openai)
    
    llm = ragas_llm(
        provider="openai",
        model="gpt-4",
        client=mock_sync_client,
        temperature=0.7
    )
    
    assert isinstance(llm, RagasLLM)
    assert llm.provider == "openai"
    assert llm.model == "gpt-4"
    assert llm.model_args.get("temperature") == 0.7


def test_unsupported_provider():
    """Test that unsupported providers raise ValueError."""
    mock_client = Mock()
    
    with pytest.raises(ValueError, match="Unsupported provider: unsupported"):
        RagasLLM(
            provider="unsupported", 
            model="test-model",
            client=mock_client
        )


def test_sync_llm_generate(mock_sync_client, monkeypatch):
    """Test sync LLM generation."""
    def mock_from_openai(client):
        return MockInstructor(client)
    
    monkeypatch.setattr('instructor.from_openai', mock_from_openai)
    
    llm = RagasLLM(
        provider="openai",
        model="gpt-4", 
        client=mock_sync_client
    )
    
    result = llm.generate("Test prompt", LLMResponseModel)
    
    assert isinstance(result, LLMResponseModel)
    assert result.response == "Instructor response"


@pytest.mark.asyncio
async def test_async_llm_agenerate(mock_async_client, monkeypatch):
    """Test async LLM generation."""
    def mock_from_openai(client):
        return MockInstructor(client)
    
    monkeypatch.setattr('instructor.from_openai', mock_from_openai)
    
    llm = RagasLLM(
        provider="openai",
        model="gpt-4",
        client=mock_async_client
    )
    
    result = await llm.agenerate("Test prompt", LLMResponseModel)
    
    assert isinstance(result, LLMResponseModel)
    assert result.response == "Instructor response"


def test_sync_client_agenerate_error(mock_sync_client, monkeypatch):
    """Test that using agenerate with sync client raises TypeError."""
    def mock_from_openai(client):
        return MockInstructor(client)
    
    monkeypatch.setattr('instructor.from_openai', mock_from_openai)
    
    llm = RagasLLM(
        provider="openai",
        model="gpt-4",
        client=mock_sync_client
    )
    
    # Test that agenerate raises TypeError with sync client
    with pytest.raises(TypeError, match="Cannot use agenerate\\(\\) with a synchronous client"):
        # Use asyncio.run to handle the coroutine
        import asyncio
        asyncio.run(llm.agenerate("Test prompt", LLMResponseModel))


def test_provider_support():
    """Test that all expected providers are supported."""
    supported_providers = ["openai", "anthropic", "cohere", "gemini", "litellm"]
    
    for provider in supported_providers:
        mock_client = Mock()
        
        # Mock the appropriate instructor function
        import instructor
        mock_instructor_func = Mock(return_value=MockInstructor(mock_client))
        setattr(instructor, f"from_{provider}", mock_instructor_func)
        
        # This should not raise an error
        try:
            llm = RagasLLM(provider=provider, model="test-model", client=mock_client)
            assert llm.provider == provider
        except Exception as e:
            pytest.fail(f"Provider {provider} should be supported but got error: {e}")


def test_llm_model_args_storage(mock_sync_client, monkeypatch):
    """Test that model arguments are properly stored."""
    def mock_from_openai(client):
        return MockInstructor(client)
    
    monkeypatch.setattr('instructor.from_openai', mock_from_openai)
    
    model_args = {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9
    }
    
    llm = RagasLLM(
        provider="openai",
        model="gpt-4",
        client=mock_sync_client,
        **model_args
    )
    
    assert llm.model_args == model_args