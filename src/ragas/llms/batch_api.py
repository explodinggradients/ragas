"""
OpenAI Batch API implementation for cost-effective evaluation.

This module provides support for OpenAI's Batch API, enabling up to 50% cost savings
for non-urgent evaluation workloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

if t.TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Batch job status enumeration."""

    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class BatchEndpoint(Enum):
    """Supported batch API endpoints."""

    CHAT_COMPLETIONS = "/v1/chat/completions"
    COMPLETIONS = "/v1/completions"
    EMBEDDINGS = "/v1/embeddings"


@dataclass
class BatchRequest:
    """Represents a single batch request."""

    custom_id: str
    method: str = "POST"
    url: str = BatchEndpoint.CHAT_COMPLETIONS.value
    body: t.Dict[str, t.Any] = field(default_factory=dict)


@dataclass
class BatchResponse:
    """Represents a single batch response."""

    id: str
    custom_id: str
    response: t.Optional[t.Dict[str, t.Any]] = None
    error: t.Optional[t.Dict[str, t.Any]] = None


class BatchJob:
    """Represents an OpenAI batch job."""

    def __init__(
        self,
        client: t.Union[OpenAI, AsyncOpenAI],
        batch_id: str,
        endpoint: str = BatchEndpoint.CHAT_COMPLETIONS.value,
        completion_window: str = "24h",
    ):
        self.client = client
        self.batch_id = batch_id
        self.endpoint = endpoint
        self.completion_window = completion_window
        self._is_async = self._check_async_client(client)

    def _check_async_client(self, client: t.Any) -> bool:
        """Check if the client is async."""
        return hasattr(client, "__aenter__") or "Async" in client.__class__.__name__

    def get_status(self) -> BatchStatus:
        """Get the current status of the batch job."""
        if self._is_async:
            raise RuntimeError("Use aget_status() for async clients")

        batch = self.client.batches.retrieve(self.batch_id)  # type: ignore[misc]
        return BatchStatus(batch.status)  # type: ignore[misc]

    async def aget_status(self) -> BatchStatus:
        """Asynchronously get the current status of the batch job."""
        if not self._is_async:
            raise RuntimeError("Use get_status() for sync clients")

        batch = await self.client.batches.retrieve(self.batch_id)  # type: ignore[misc]
        return BatchStatus(batch.status)  # type: ignore[misc]

    def wait_for_completion(
        self, poll_interval: float = 30.0, timeout: float = 86400.0
    ) -> BatchStatus:
        """Wait for batch job completion with polling."""
        if self._is_async:
            raise RuntimeError("Use await_completion() for async clients")

        start_time = time.time()
        while True:
            status = self.get_status()

            if status in [
                BatchStatus.COMPLETED,
                BatchStatus.FAILED,
                BatchStatus.EXPIRED,
                BatchStatus.CANCELLED,
            ]:
                return status

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Batch job {self.batch_id} did not complete within {timeout} seconds"
                )

            logger.info(
                f"Batch job {self.batch_id} status: {status.value}. Waiting {poll_interval}s..."
            )
            time.sleep(poll_interval)

    async def await_completion(
        self, poll_interval: float = 30.0, timeout: float = 86400.0
    ) -> BatchStatus:
        """Asynchronously wait for batch job completion with polling."""
        if not self._is_async:
            raise RuntimeError("Use wait_for_completion() for sync clients")

        start_time = time.time()
        while True:
            status = await self.aget_status()

            if status in [
                BatchStatus.COMPLETED,
                BatchStatus.FAILED,
                BatchStatus.EXPIRED,
                BatchStatus.CANCELLED,
            ]:
                return status

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Batch job {self.batch_id} did not complete within {timeout} seconds"
                )

            logger.info(
                f"Batch job {self.batch_id} status: {status.value}. Waiting {poll_interval}s..."
            )
            await asyncio.sleep(poll_interval)

    def get_results(self) -> t.List[BatchResponse]:
        """Retrieve and parse batch job results."""
        if self._is_async:
            raise RuntimeError("Use aget_results() for async clients")

        batch = self.client.batches.retrieve(self.batch_id)  # type: ignore[misc]

        if batch.status != "completed":  # type: ignore[misc]
            raise ValueError(
                f"Batch job {self.batch_id} is not completed. Status: {batch.status}"  # type: ignore[misc]
            )

        if not batch.output_file_id:  # type: ignore[misc]
            raise ValueError(f"Batch job {self.batch_id} has no output file")

        # Download and parse results
        result_content = self.client.files.content(batch.output_file_id).content  # type: ignore[misc]
        return self._parse_results(result_content)

    async def aget_results(self) -> t.List[BatchResponse]:
        """Asynchronously retrieve and parse batch job results."""
        if not self._is_async:
            raise RuntimeError("Use get_results() for sync clients")

        batch = await self.client.batches.retrieve(self.batch_id)  # type: ignore[misc]

        if batch.status != "completed":  # type: ignore[misc]
            raise ValueError(
                f"Batch job {self.batch_id} is not completed. Status: {batch.status}"  # type: ignore[misc]
            )

        if not batch.output_file_id:  # type: ignore[misc]
            raise ValueError(f"Batch job {self.batch_id} has no output file")

        # Download and parse results
        result_content = await self.client.files.content(batch.output_file_id)  # type: ignore[misc]
        return self._parse_results(result_content.content)

    def _parse_results(self, content: bytes) -> t.List[BatchResponse]:
        """Parse batch results from JSONL content."""
        results = []
        for line in content.decode("utf-8").strip().split("\n"):
            if line.strip():
                result_data = json.loads(line)
                results.append(
                    BatchResponse(
                        id=result_data["id"],
                        custom_id=result_data["custom_id"],
                        response=result_data.get("response"),
                        error=result_data.get("error"),
                    )
                )
        return results


class OpenAIBatchAPI:
    """OpenAI Batch API client wrapper."""

    def __init__(
        self,
        client: t.Union[OpenAI, AsyncOpenAI],
        max_batch_size: int = 50000,
        max_file_size_mb: int = 100,
    ):
        self.client = client
        self.max_batch_size = max_batch_size
        self.max_file_size_mb = max_file_size_mb
        self._is_async = self._check_async_client(client)

    def _check_async_client(self, client: t.Any) -> bool:
        """Check if the client is async."""
        return hasattr(client, "__aenter__") or "Async" in client.__class__.__name__

    def _create_jsonl_content(self, requests: t.List[BatchRequest]) -> str:
        """Create JSONL content from batch requests."""
        lines = []
        for request in requests:
            line = json.dumps(
                {
                    "custom_id": request.custom_id,
                    "method": request.method,
                    "url": request.url,
                    "body": request.body,
                }
            )
            lines.append(line)
        return "\n".join(lines)

    def _validate_requests(self, requests: t.List[BatchRequest]) -> None:
        """Validate batch requests."""
        if len(requests) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(requests)} exceeds maximum {self.max_batch_size}"
            )

        # Check for duplicate custom_ids
        custom_ids = [req.custom_id for req in requests]
        if len(custom_ids) != len(set(custom_ids)):
            raise ValueError("Duplicate custom_id values found in batch requests")

        # Estimate file size (rough approximation)
        jsonl_content = self._create_jsonl_content(requests)
        size_mb = len(jsonl_content.encode("utf-8")) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValueError(
                f"Batch file size {size_mb:.2f}MB exceeds maximum {self.max_file_size_mb}MB"
            )

    def create_batch(
        self,
        requests: t.List[BatchRequest],
        endpoint: str = BatchEndpoint.CHAT_COMPLETIONS.value,
        completion_window: str = "24h",
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> BatchJob:
        """Create a new batch job."""
        if self._is_async:
            raise RuntimeError("Use acreate_batch() for async clients")

        self._validate_requests(requests)

        # Create JSONL file
        jsonl_content = self._create_jsonl_content(requests)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            temp_file_path = f.name

        try:
            # Upload file
            with open(temp_file_path, "rb") as f:
                batch_file = self.client.files.create(file=f, purpose="batch")  # type: ignore[misc]

            # Create batch job
            batch_job = self.client.batches.create(  # type: ignore[misc]
                input_file_id=batch_file.id,  # type: ignore[misc]
                endpoint=endpoint,  # type: ignore[arg-type]
                completion_window=completion_window,  # type: ignore[arg-type]
                metadata=metadata or {},
            )

            return BatchJob(
                client=self.client,
                batch_id=batch_job.id,  # type: ignore[misc]
                endpoint=endpoint,  # type: ignore[arg-type]
                completion_window=completion_window,  # type: ignore[arg-type]
            )

        finally:
            # Clean up temp file
            Path(temp_file_path).unlink(missing_ok=True)

    async def acreate_batch(
        self,
        requests: t.List[BatchRequest],
        endpoint: str = BatchEndpoint.CHAT_COMPLETIONS.value,
        completion_window: str = "24h",
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> BatchJob:
        """Asynchronously create a new batch job."""
        if not self._is_async:
            raise RuntimeError("Use create_batch() for sync clients")

        self._validate_requests(requests)

        # Create JSONL file
        jsonl_content = self._create_jsonl_content(requests)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            temp_file_path = f.name

        try:
            # Upload file
            with open(temp_file_path, "rb") as f:
                batch_file = await self.client.files.create(file=f, purpose="batch")  # type: ignore[misc]

            # Create batch job
            batch_job = await self.client.batches.create(  # type: ignore[misc]
                input_file_id=batch_file.id,  # type: ignore[misc]
                endpoint=endpoint,  # type: ignore[arg-type]
                completion_window=completion_window,  # type: ignore[arg-type]
                metadata=metadata or {},
            )

            return BatchJob(
                client=self.client,
                batch_id=batch_job.id,  # type: ignore[misc]
                endpoint=endpoint,  # type: ignore[arg-type]
                completion_window=completion_window,  # type: ignore[arg-type]
            )

        finally:
            # Clean up temp file
            Path(temp_file_path).unlink(missing_ok=True)

    def create_chat_completion_requests(
        self,
        prompts: t.List[t.Dict[str, t.Any]],
        model: str,
        **kwargs: t.Any,
    ) -> t.List[BatchRequest]:
        """Create batch requests for chat completions."""
        requests = []
        for i, prompt_data in enumerate(prompts):
            request = BatchRequest(
                custom_id=f"request-{i}",
                url=BatchEndpoint.CHAT_COMPLETIONS.value,
                body={
                    "model": model,
                    "messages": prompt_data.get("messages", []),
                    **kwargs,
                },
            )
            # Allow custom_id override
            if "custom_id" in prompt_data:
                request.custom_id = prompt_data["custom_id"]

            requests.append(request)

        return requests


def create_batch_api(client: t.Union[OpenAI, AsyncOpenAI]) -> OpenAIBatchAPI:
    """Factory function to create OpenAI Batch API instance."""
    return OpenAIBatchAPI(client)
