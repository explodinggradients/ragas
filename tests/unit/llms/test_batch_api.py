"""
Unit tests for OpenAI Batch API functionality.
"""

import json
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from ragas.llms.batch_api import (
    BatchEndpoint,
    BatchJob,
    BatchRequest,
    BatchResponse,
    BatchStatus,
    OpenAIBatchAPI,
    create_batch_api,
)


class TestBatchRequest:
    """Test BatchRequest dataclass."""

    def test_batch_request_creation(self):
        """Test creating a batch request."""
        request = BatchRequest(
            custom_id="test-1",
            method="POST",
            url=BatchEndpoint.CHAT_COMPLETIONS.value,
            body={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert request.custom_id == "test-1"
        assert request.method == "POST"
        assert request.url == "/v1/chat/completions"
        assert request.body["model"] == "gpt-4o-mini"


class TestBatchResponse:
    """Test BatchResponse dataclass."""

    def test_batch_response_creation(self):
        """Test creating a batch response."""
        response = BatchResponse(
            id="batch-123",
            custom_id="test-1",
            response={"choices": [{"message": {"content": "Hello!"}}]},
            error=None,
        )

        assert response.id == "batch-123"
        assert response.custom_id == "test-1"
        assert response.response is not None
        assert response.error is None


class TestBatchJob:
    """Test BatchJob class."""

    def test_batch_job_sync_client(self):
        """Test BatchJob with sync client."""
        mock_client = Mock()
        mock_client.__class__.__name__ = "OpenAI"

        job = BatchJob(
            client=mock_client,
            batch_id="batch-123",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
        )

        assert job.client == mock_client
        assert job.batch_id == "batch-123"
        assert job.endpoint == "/v1/chat/completions"
        assert not job._is_async

    def test_batch_job_async_client(self):
        """Test BatchJob with async client."""
        mock_client = Mock()
        mock_client.__class__.__name__ = "AsyncOpenAI"

        job = BatchJob(client=mock_client, batch_id="batch-123")

        assert job._is_async

    def test_get_status_sync(self):
        """Test getting batch status synchronously."""
        mock_client = Mock()
        mock_client.__class__.__name__ = "OpenAI"
        mock_batch = Mock()
        mock_batch.status = "completed"
        mock_client.batches.retrieve.return_value = mock_batch

        job = BatchJob(client=mock_client, batch_id="batch-123")
        status = job.get_status()

        assert status == BatchStatus.COMPLETED
        mock_client.batches.retrieve.assert_called_once_with("batch-123")

    @pytest.mark.asyncio
    async def test_get_status_async(self):
        """Test getting batch status asynchronously."""
        mock_client = AsyncMock()
        mock_client.__class__.__name__ = "AsyncOpenAI"
        mock_batch = Mock()
        mock_batch.status = "in_progress"
        mock_client.batches.retrieve.return_value = mock_batch

        job = BatchJob(client=mock_client, batch_id="batch-123")
        status = await job.aget_status()

        assert status == BatchStatus.IN_PROGRESS
        mock_client.batches.retrieve.assert_called_once_with("batch-123")

    def test_parse_results(self):
        """Test parsing batch results from JSONL content."""
        mock_client = Mock()
        mock_client.__class__.__name__ = "OpenAI"

        job = BatchJob(client=mock_client, batch_id="batch-123")

        # Mock JSONL content
        jsonl_content = """{"id": "batch-123", "custom_id": "req-1", "response": {"choices": [{"message": {"content": "Hello"}}]}}
{"id": "batch-123", "custom_id": "req-2", "error": {"message": "Rate limit exceeded"}}"""

        results = job._parse_results(jsonl_content.encode("utf-8"))

        assert len(results) == 2
        assert results[0].custom_id == "req-1"
        assert results[0].response is not None
        assert results[0].error is None
        assert results[1].custom_id == "req-2"
        assert results[1].response is None
        assert results[1].error is not None


class TestOpenAIBatchAPI:
    """Test OpenAIBatchAPI class."""

    def test_batch_api_creation(self):
        """Test creating batch API instance."""
        mock_client = Mock()
        mock_client.__class__.__name__ = "OpenAI"

        api = OpenAIBatchAPI(client=mock_client)

        assert api.client == mock_client
        assert api.max_batch_size == 50000
        assert api.max_file_size_mb == 100
        assert not api._is_async

    def test_create_jsonl_content(self):
        """Test creating JSONL content from requests."""
        mock_client = Mock()
        api = OpenAIBatchAPI(client=mock_client)

        requests = [
            BatchRequest(
                custom_id="req-1",
                body={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            ),
            BatchRequest(
                custom_id="req-2",
                body={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            ),
        ]

        jsonl_content = api._create_jsonl_content(requests)
        lines = jsonl_content.split("\n")

        assert len(lines) == 2
        req1 = json.loads(lines[0])
        assert req1["custom_id"] == "req-1"
        assert req1["method"] == "POST"
        assert req1["url"] == "/v1/chat/completions"

    def test_validate_requests_success(self):
        """Test successful request validation."""
        mock_client = Mock()
        api = OpenAIBatchAPI(client=mock_client, max_batch_size=10)

        requests = [
            BatchRequest(custom_id="req-1", body={"model": "gpt-4o-mini"}),
            BatchRequest(custom_id="req-2", body={"model": "gpt-4o-mini"}),
        ]

        # Should not raise any exception
        api._validate_requests(requests)

    def test_validate_requests_too_many(self):
        """Test validation failure for too many requests."""
        mock_client = Mock()
        api = OpenAIBatchAPI(client=mock_client, max_batch_size=1)

        requests = [
            BatchRequest(custom_id="req-1", body={}),
            BatchRequest(custom_id="req-2", body={}),
        ]

        with pytest.raises(ValueError, match="Batch size 2 exceeds maximum 1"):
            api._validate_requests(requests)

    def test_validate_requests_duplicate_ids(self):
        """Test validation failure for duplicate custom IDs."""
        mock_client = Mock()
        api = OpenAIBatchAPI(client=mock_client)

        requests = [
            BatchRequest(custom_id="req-1", body={}),
            BatchRequest(custom_id="req-1", body={}),  # Duplicate ID
        ]

        with pytest.raises(ValueError, match="Duplicate custom_id values found"):
            api._validate_requests(requests)

    def test_create_chat_completion_requests(self):
        """Test creating chat completion requests."""
        mock_client = Mock()
        api = OpenAIBatchAPI(client=mock_client)

        prompts = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "Hi"}], "custom_id": "custom-1"},
        ]

        requests = api.create_chat_completion_requests(
            prompts=prompts, model="gpt-4o-mini", temperature=0.7
        )

        assert len(requests) == 2
        assert requests[0].custom_id == "request-0"
        assert requests[1].custom_id == "custom-1"
        assert requests[0].body["model"] == "gpt-4o-mini"
        assert requests[0].body["temperature"] == 0.7

    @patch("tempfile.NamedTemporaryFile")
    @patch("builtins.open", mock_open())
    @patch("pathlib.Path.unlink")
    def test_create_batch_sync(self, mock_unlink, mock_temp_file):
        """Test creating batch job synchronously."""
        # Setup mocks
        mock_client = Mock()
        mock_client.__class__.__name__ = "OpenAI"

        mock_temp_file_instance = Mock()
        mock_temp_file_instance.name = "/tmp/test.jsonl"
        mock_temp_file_instance.__enter__ = Mock(return_value=mock_temp_file_instance)
        mock_temp_file_instance.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp_file_instance

        mock_file_obj = Mock()
        mock_file_obj.id = "file-123"
        mock_client.files.create.return_value = mock_file_obj

        mock_batch_job = Mock()
        mock_batch_job.id = "batch-123"
        mock_client.batches.create.return_value = mock_batch_job

        api = OpenAIBatchAPI(client=mock_client)
        requests = [BatchRequest(custom_id="req-1", body={"model": "gpt-4o-mini"})]

        batch_job = api.create_batch(requests=requests)

        assert batch_job.batch_id == "batch-123"
        mock_client.files.create.assert_called_once()
        mock_client.batches.create.assert_called_once()


def test_create_batch_api():
    """Test factory function."""
    mock_client = Mock()
    api = create_batch_api(mock_client)

    assert isinstance(api, OpenAIBatchAPI)
    assert api.client == mock_client
