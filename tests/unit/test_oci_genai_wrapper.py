"""Tests for OCI Gen AI wrapper."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from langchain_core.prompt_values import StringPromptValue
from langchain_core.outputs import LLMResult, Generation

from ragas.llms.oci_genai_wrapper import OCIGenAIWrapper, oci_genai_factory


class TestOCIGenAIWrapper:
    """Test cases for OCI Gen AI wrapper."""

    @pytest.fixture
    def mock_oci_client(self):
        """Mock OCI client for testing."""
        with patch('ragas.llms.oci_genai_wrapper.GenerativeAiClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def oci_wrapper(self, mock_oci_client):
        """Create OCI wrapper instance for testing."""
        return OCIGenAIWrapper(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example"
        )

    def test_initialization(self, mock_oci_client):
        """Test OCI wrapper initialization."""
        wrapper = OCIGenAIWrapper(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example"
        )
        
        assert wrapper.model_id == "cohere.command"
        assert wrapper.compartment_id == "ocid1.compartment.oc1..example"
        assert wrapper.client == mock_oci_client

    def test_initialization_with_endpoint(self, mock_oci_client):
        """Test OCI wrapper initialization with endpoint."""
        wrapper = OCIGenAIWrapper(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example",
            endpoint_id="ocid1.endpoint.oc1..example"
        )
        
        assert wrapper.endpoint_id == "ocid1.endpoint.oc1..example"

    def test_convert_prompt_to_string(self, oci_wrapper):
        """Test prompt conversion to string."""
        prompt = StringPromptValue(text="Hello, world!")
        result = oci_wrapper._convert_prompt_to_string(prompt)
        assert result == "Hello, world!"

    def test_create_generation_request(self, oci_wrapper):
        """Test generation request creation."""
        request = oci_wrapper._create_generation_request(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=100,
            stop=["stop"]
        )
        
        assert request["compartment_id"] == oci_wrapper.compartment_id
        assert request["serving_mode"]["model_id"] == oci_wrapper.model_id
        assert request["inference_request"]["messages"][0]["content"] == "Test prompt"
        assert request["inference_request"]["temperature"] == 0.5
        assert request["inference_request"]["max_tokens"] == 100
        assert request["inference_request"]["stop"] == ["stop"]

    def test_create_generation_request_with_endpoint(self):
        """Test generation request creation with endpoint."""
        wrapper = OCIGenAIWrapper(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example",
            endpoint_id="ocid1.endpoint.oc1..example"
        )
        
        request = wrapper._create_generation_request("Test prompt")
        assert request["serving_mode"]["endpoint_id"] == "ocid1.endpoint.oc1..example"

    def test_generate_text(self, oci_wrapper, mock_oci_client):
        """Test synchronous text generation."""
        # Mock response
        mock_response = Mock()
        mock_response.data.choices = [Mock()]
        mock_response.data.choices[0].message.content = "Generated text"
        mock_oci_client.generate_text.return_value = mock_response
        
        prompt = StringPromptValue(text="Test prompt")
        result = oci_wrapper.generate_text(prompt, n=1, temperature=0.5)
        
        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        assert result.generations[0][0].text == "Generated text"
        
        # Verify client was called
        mock_oci_client.generate_text.assert_called_once()

    def test_generate_text_multiple_completions(self, oci_wrapper, mock_oci_client):
        """Test multiple completions generation."""
        # Mock response
        mock_response = Mock()
        mock_response.data.choices = [Mock()]
        mock_response.data.choices[0].message.content = "Generated text"
        mock_oci_client.generate_text.return_value = mock_response
        
        prompt = StringPromptValue(text="Test prompt")
        result = oci_wrapper.generate_text(prompt, n=3, temperature=0.5)
        
        assert isinstance(result, LLMResult)
        assert len(result.generations) == 3
        assert mock_oci_client.generate_text.call_count == 3

    @pytest.mark.asyncio
    async def test_agenerate_text(self, oci_wrapper, mock_oci_client):
        """Test asynchronous text generation."""
        # Mock response
        mock_response = Mock()
        mock_response.data.choices = [Mock()]
        mock_response.data.choices[0].message.content = "Generated text"
        mock_oci_client.generate_text.return_value = mock_response
        
        prompt = StringPromptValue(text="Test prompt")
        result = await oci_wrapper.agenerate_text(prompt, n=1, temperature=0.5)
        
        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        assert result.generations[0][0].text == "Generated text"

    def test_is_finished(self, oci_wrapper):
        """Test is_finished method."""
        # Test with valid generations
        generations = [[Generation(text="Valid text")]]
        result = LLMResult(generations=generations)
        assert oci_wrapper.is_finished(result) is True
        
        # Test with empty text
        generations = [[Generation(text="")]]
        result = LLMResult(generations=generations)
        assert oci_wrapper.is_finished(result) is False
        
        # Test with whitespace only
        generations = [[Generation(text="   ")]]
        result = LLMResult(generations=generations)
        assert oci_wrapper.is_finished(result) is False

    def test_repr(self, oci_wrapper):
        """Test string representation."""
        repr_str = repr(oci_wrapper)
        assert "OCIGenAIWrapper" in repr_str
        assert "cohere.command" in repr_str
        assert "ocid1.compartment.oc1..example" in repr_str

    def test_import_error(self):
        """Test import error when OCI SDK is not available."""
        with patch('ragas.llms.oci_genai_wrapper.oci', None):
            with pytest.raises(ImportError, match="OCI SDK not found"):
                OCIGenAIWrapper(
                    model_id="cohere.command",
                    compartment_id="ocid1.compartment.oc1..example"
                )


class TestOCIGenAIFactory:
    """Test cases for OCI Gen AI factory function."""

    @patch('ragas.llms.oci_genai_wrapper.OCIGenAIWrapper')
    def test_oci_genai_factory(self, mock_wrapper_class):
        """Test OCI Gen AI factory function."""
        mock_wrapper = Mock()
        mock_wrapper_class.return_value = mock_wrapper
        
        result = oci_genai_factory(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example",
            endpoint_id="ocid1.endpoint.oc1..example"
        )
        
        mock_wrapper_class.assert_called_once_with(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example",
            endpoint_id="ocid1.endpoint.oc1..example",
            config=None,
            run_config=None,
            cache=None
        )
        assert result == mock_wrapper

    @patch('ragas.llms.oci_genai_wrapper.OCIGenAIWrapper')
    def test_oci_genai_factory_with_config(self, mock_wrapper_class):
        """Test OCI Gen AI factory with custom config."""
        config = {"user": "test_user", "key_file": "test_key.pem"}
        
        oci_genai_factory(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example",
            config=config
        )
        
        mock_wrapper_class.assert_called_once_with(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example",
            endpoint_id=None,
            config=config,
            run_config=None,
            cache=None
        )
