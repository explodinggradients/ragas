from __future__ import annotations

import typing as t
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from ragas._analytics import LLMUsageEvent
from ragas.llms.litellm_llm import LiteLLMStructuredLLM


class _Out(BaseModel):
    text: str


class _StubSyncClient:
    """Sync client that satisfies LiteLLMStructuredLLM._check_client_async's sync path."""

    class chat:
        class completions:
            @staticmethod
            def create(*, model, messages, response_model, **kwargs):
                return response_model(text="sync")


class _StubAsyncClient:
    """Async client that satisfies _check_client_async via the chat.completions.create branch."""

    class chat:
        class completions:
            @staticmethod
            async def create(*, model, messages, response_model, **kwargs):
                return response_model(text="async")


def _make_sync_llm() -> LiteLLMStructuredLLM:
    return LiteLLMStructuredLLM(
        client=_StubSyncClient(),
        model="test-model",
        provider="test",
    )


def _make_async_llm() -> LiteLLMStructuredLLM:
    return LiteLLMStructuredLLM(
        client=_StubAsyncClient(),
        model="test-model",
        provider="test",
    )


@pytest.fixture
def captured_events():
    """Patch litellm_llm.track to capture every LLMUsageEvent without forwarding."""
    captured: list = []

    def fake_track(event: LLMUsageEvent) -> None:
        captured.append(event)

    with patch("ragas.llms.litellm_llm.track", side_effect=fake_track):
        yield captured


def _is_litellm_usage_event(ev: t.Any) -> bool:
    return isinstance(ev, LLMUsageEvent) and ev.llm_type == "litellm"


def test_generate_on_sync_client_tracks_once(captured_events):
    """Sync path: generate() should emit exactly one LLMUsageEvent."""
    llm = _make_sync_llm()
    assert llm.is_async is False

    llm.generate(prompt="hello", response_model=_Out)

    events = [e for e in captured_events if _is_litellm_usage_event(e)]
    assert len(events) == 1
    assert events[0].is_async is False
    assert events[0].num_requests == 1
    assert events[0].provider == "test"
    assert events[0].model == "test-model"


def test_generate_on_async_client_tracks_once_not_twice(captured_events):
    """Async path: generate() routes through agenerate(); both must not double-track.

    Before the fix, captured_events would contain two LLMUsageEvent entries
    for a single generate() call on an async client (one from agenerate, one
    from the tail of generate). After the fix, exactly one.
    """
    llm = _make_async_llm()
    assert llm.is_async is True

    llm.generate(prompt="hello", response_model=_Out)

    events = [e for e in captured_events if _is_litellm_usage_event(e)]
    assert len(events) == 1
    assert events[0].is_async is True
    assert events[0].num_requests == 1
    assert events[0].provider == "test"
    assert events[0].model == "test-model"


@pytest.mark.asyncio
async def test_agenerate_on_async_client_tracks_once(captured_events):
    """Direct agenerate() call: exactly one LLMUsageEvent."""
    llm = _make_async_llm()
    assert llm.is_async is True

    await llm.agenerate(prompt="hello", response_model=_Out)

    events = [e for e in captured_events if _is_litellm_usage_event(e)]
    assert len(events) == 1
    assert events[0].is_async is True
    assert events[0].num_requests == 1
