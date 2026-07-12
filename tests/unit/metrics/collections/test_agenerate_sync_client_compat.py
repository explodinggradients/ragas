"""Regression tests for the collections metrics sync-LLM async dispatch.

See https://github.com/vibrantlabsai/ragas/issues/2599.

Before the fix, every ``await self.llm.agenerate(...)`` call inside
collections metrics raised ``TypeError: Cannot use agenerate() with a
synchronous client`` when the user supplied a sync ``llm_factory(...)`` client
(sync ``OpenAI``, sync ``Anthropic``, ``google-genai`` ``Client``,
LiteLLM with a sync create function, etc).

The fix routes those calls through ``ragas.metrics.collections._async_compat
.call_llm_async``, which uses ``asyncio.to_thread`` to run the sync
``InstructorLLM.generate`` off the event loop while keeping the async
``InstructorLLM.agenerate`` path intact.
"""

from __future__ import annotations

import asyncio
import typing as t
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from ragas.llms.base import InstructorLLM, llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    ContextEntityRecall,
    ContextPrecision,
    ContextRelevance,
)


class _OutputA(BaseModel):
    statements: list[str] = ["ok"]


class _OutputB(BaseModel):
    TP: list[str] = []
    FP: list[str] = []
    FN: list[str] = ["x"]
    reason: str = "r"


class _ContextPrecisionOutput(BaseModel):
    reason: str = "r"
    verdict: int = 1


class _ContextRelevanceOutput(BaseModel):
    reason: str = "r"
    rating: int = 1


class _EntitiesList(BaseModel):
    entities: list[str] = ["e1"]


class _MockAsyncCreate:
    """Async callable that returns the requested response model."""

    def __init__(self, *responses: BaseModel) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def __call__(self, *args: t.Any, **kwargs: t.Any) -> BaseModel:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        # Fallback: return the last response so repeated calls don't crash.
        return self.responses[-1]


class _MockSyncCreate:
    """Sync callable that returns the requested response model."""

    def __init__(self, *responses: BaseModel) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> BaseModel:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return self.responses[-1]


def _build_mock_instructor(
    client: Mock, outputs: t.Sequence[BaseModel], is_async: bool = False
):
    """Return an Instructor-like wrapper whose ``__class__.__name__`` matches
    ``"Instructor"`` (sync) or ``"AsyncInstructor"`` (async) so the existing
    ``_check_client_async`` heuristic in ``InstructorLLM`` recognises it.

    ``outputs`` is the sequence of mock responses to return in order; pass
    multiple when a metric invokes ``llm.generate`` / ``llm.agenerate`` more
    than once with different response_model types.
    """

    class _Base:
        # Loose attributes are set below to mirror the real Instructor wrapper
        # shape (``client``, ``chat.completions.create``, ``messages.create``).
        client: t.Any
        chat: t.Any
        messages: t.Any

    target_name = "AsyncInstructor" if is_async else "Instructor"
    _Base.__name__ = target_name
    wrapper = _Base()
    wrapper.client = client
    wrapper.chat = Mock()
    wrapper.chat.completions = Mock()
    wrapper.messages = Mock()
    if is_async:
        wrapper.chat.completions.create = _MockAsyncCreate(*outputs)
    else:
        wrapper.chat.completions.create = _MockSyncCreate(*outputs)
    wrapper.messages.create = wrapper.chat.completions.create
    return wrapper


def _build_sync_llm(
    monkeypatch: pytest.MonkeyPatch, *outputs: BaseModel
) -> "InstructorLLM":
    """Build a real ``InstructorLLM`` whose wrapped client is sync."""

    client = Mock()
    client.is_async = False
    outputs_list = list(outputs)

    def factory(c, mode=None):
        c.is_async = False
        return _build_mock_instructor(c, outputs_list, is_async=False)

    monkeypatch.setattr("ragas.llms.base.instructor.from_openai", factory)
    llm = llm_factory("gpt-4o-mini", provider="openai", client=client)
    assert isinstance(llm, InstructorLLM)
    assert llm.is_async is False
    return llm


def _build_async_llm(
    monkeypatch: pytest.MonkeyPatch, *outputs: BaseModel
) -> "InstructorLLM":
    """Build a real ``InstructorLLM`` whose wrapped client is async."""

    client = Mock()
    client.is_async = True
    outputs_list = list(outputs)

    def factory(c, mode=None):
        c.is_async = True
        return _build_mock_instructor(c, outputs_list, is_async=True)

    monkeypatch.setattr("ragas.llms.base.instructor.from_openai", factory)
    llm = llm_factory("gpt-4o-mini", provider="openai", client=client)
    assert isinstance(llm, InstructorLLM)
    assert llm.is_async is True
    return llm


def test_answer_correctness_ascore_routes_sync_llm(monkeypatch):
    """Sync InstructorLLM should no longer break collections.ascore (#2599)."""

    # AnswerCorrectness makes two LLM calls: one for response statements, one
    # for reference statements, then a third for classification. Provide three
    # matching mock responses in order.
    llm = _build_sync_llm(
        monkeypatch,
        _OutputA(),  # response statements
        _OutputA(),  # reference statements
        _OutputB(),  # classification (one FN -> recall 0, precision 0 -> f1 0)
    )

    metric = AnswerCorrectness(llm=llm, weights=[1.0, 0.0])

    result = asyncio.run(
        metric.ascore(
            user_input="What is the capital of France?",
            response="Paris",
            reference="Paris is the capital of France.",
        )
    )

    # OutputB has TP=[], FP=[], FN=["x"], so factuality_score -> 0.0.
    assert result.value == 0.0


@pytest.mark.asyncio
async def test_answer_correctness_ascore_async_llm(monkeypatch):
    """The async path still works for true async InstructorLLMs."""

    llm = _build_async_llm(
        monkeypatch,
        _OutputA(),
        _OutputA(),
        _OutputB(),
    )

    metric = AnswerCorrectness(llm=llm, weights=[1.0, 0.0])

    result = await metric.ascore(
        user_input="What is the capital of France?",
        response="Paris",
        reference="Paris is the capital of France.",
    )

    assert result.value == 0.0


def test_context_precision_routes_sync_llm(monkeypatch):
    """ContextPrecision.ascore should also work against sync InstructorLLMs."""

    # ContextPrecision (with reference) iterates over retrieved_contexts, one
    # LLM call per context. Default to a single context to keep the test small.
    llm = _build_sync_llm(monkeypatch, _ContextPrecisionOutput())

    metric = ContextPrecision(llm=llm)

    result = asyncio.run(
        metric.ascore(
            user_input="Q?",
            reference="A.",
            retrieved_contexts=["ctx"],
        )
    )

    assert 0.0 <= result.value <= 1.0


def test_context_relevance_routes_sync_llm(monkeypatch):
    """ContextRelevance.ascore should also work against sync InstructorLLMs."""

    # ContextRelevance uses two judges per context; provide two mock outputs.
    llm = _build_sync_llm(
        monkeypatch, _ContextRelevanceOutput(), _ContextRelevanceOutput()
    )

    metric = ContextRelevance(llm=llm)

    result = asyncio.run(
        metric.ascore(
            user_input="Q?",
            retrieved_contexts=["ctx"],
        )
    )

    assert 0.0 <= result.value <= 1.0


def test_context_entity_recall_routes_sync_llm(monkeypatch):
    """ContextEntityRecall.ascore should also work against sync InstructorLLMs."""

    # ContextEntityRecall makes one LLM call per context to extract entities.
    llm = _build_sync_llm(monkeypatch, _EntitiesList())

    metric = ContextEntityRecall(llm=llm)

    result = asyncio.run(
        metric.ascore(
            reference="Some reference text with Apple and Banana.",
            retrieved_contexts=["ctx"],
        )
    )

    assert 0.0 <= result.value <= 1.0


@pytest.mark.asyncio
async def test_call_llm_async_routes_async_client(monkeypatch):
    """The helper itself routes async InstructorLLMs through agenerate."""

    llm = _build_async_llm(monkeypatch, _OutputA())
    from ragas.metrics.collections._async_compat import call_llm_async

    result = await call_llm_async(llm, "hi", _OutputA)
    assert isinstance(result, _OutputA)


def test_call_llm_async_routes_sync_client_via_to_thread(monkeypatch):
    """The helper itself routes sync InstructorLLMs through asyncio.to_thread."""

    llm = _build_sync_llm(monkeypatch, _OutputA())
    from ragas.metrics.collections._async_compat import call_llm_async

    result = asyncio.run(call_llm_async(llm, "hi", _OutputA))
    assert isinstance(result, _OutputA)
