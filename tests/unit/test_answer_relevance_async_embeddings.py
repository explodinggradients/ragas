from __future__ import annotations

from threading import get_ident
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from ragas.metrics._answer_relevance import ResponseRelevanceOutput, ResponseRelevancy


def _answers() -> list[ResponseRelevanceOutput]:
    return [
        ResponseRelevanceOutput(question=f"generated question {index}", noncommittal=0)
        for index in range(3)
    ]


def _metric(embeddings: Any) -> ResponseRelevancy:
    return ResponseRelevancy(llm=cast(Any, object()), embeddings=embeddings)


class _LegacyAsyncEmbeddings:
    def __init__(self) -> None:
        self.async_query_calls: list[str] = []
        self.async_document_calls: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("the blocking query method should not be used")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the blocking document method should not be used")

    async def aembed_query(self, text: str) -> list[float]:
        self.async_query_calls.append(text)
        return [1.0, 0.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.async_document_calls.append(texts)
        return [[1.0, 0.0] for _ in texts]


class _ModernAsyncEmbeddings:
    def __init__(self) -> None:
        self.async_text_calls: list[str] = []
        self.async_texts_calls: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        raise AssertionError("the blocking text method should not be used")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the blocking texts method should not be used")

    async def aembed_text(self, text: str) -> list[float]:
        self.async_text_calls.append(text)
        return [1.0, 0.0]

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        self.async_texts_calls.append(texts)
        return [[1.0, 0.0] for _ in texts]


class _SyncOnlyEmbeddings:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def embed_query(self, text: str) -> list[float]:
        self.thread_ids.append(get_ident())
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.thread_ids.append(get_ident())
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_ascore_uses_legacy_async_embedding_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = _LegacyAsyncEmbeddings()
    metric = _metric(embeddings)
    monkeypatch.setattr(
        metric.question_generation,
        "generate_multiple",
        AsyncMock(return_value=_answers()),
    )

    score = await metric._ascore(
        {"user_input": "original question", "response": "answer"}, callbacks=[]
    )

    assert score == pytest.approx(1.0)
    assert embeddings.async_query_calls == ["original question"]
    assert embeddings.async_document_calls == [
        ["generated question 0", "generated question 1", "generated question 2"]
    ]


@pytest.mark.asyncio
async def test_ascore_uses_modern_async_embedding_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = _ModernAsyncEmbeddings()
    metric = _metric(embeddings)
    monkeypatch.setattr(
        metric.question_generation,
        "generate_multiple",
        AsyncMock(return_value=_answers()),
    )

    score = await metric._ascore(
        {"user_input": "original question", "response": "answer"}, callbacks=[]
    )

    assert score == pytest.approx(1.0)
    assert embeddings.async_text_calls == ["original question"]
    assert embeddings.async_texts_calls == [
        ["generated question 0", "generated question 1", "generated question 2"]
    ]


@pytest.mark.asyncio
async def test_ascore_offloads_sync_only_embedding_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = _SyncOnlyEmbeddings()
    metric = _metric(embeddings)
    monkeypatch.setattr(
        metric.question_generation,
        "generate_multiple",
        AsyncMock(return_value=_answers()),
    )
    event_loop_thread = get_ident()

    score = await metric._ascore(
        {"user_input": "original question", "response": "answer"}, callbacks=[]
    )

    assert score == pytest.approx(1.0)
    assert len(embeddings.thread_ids) == 2
    assert all(thread_id != event_loop_thread for thread_id in embeddings.thread_ids)
