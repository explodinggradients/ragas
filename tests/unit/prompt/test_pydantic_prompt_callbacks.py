"""Regression tests for callback propagation through Instructor LLM calls.

See https://github.com/vibrantlabsai/ragas/issues/2575.

Before the fix, ``InstructorLLM.generate()`` / ``agenerate()`` bypassed the
LangChain callback handler. ``PydanticPrompt.generate_multiple`` reached the
Instructor LLM and constructed an ``LLMResult`` without ever driving the
``Callbacks`` that callers such as ``CostCallbackHandler`` rely on to populate
``token_usage_parser`` output.

These tests exercise a mock Instructor pipeline end-to-end so the regression
stays detectable without needing real provider credentials.
"""

from __future__ import annotations

import asyncio
import typing as t
from unittest.mock import Mock

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel

from ragas.cost import CostCallbackHandler, TokenUsage
from ragas.llms.base import InstructorBaseRagasLLM, InstructorLLM, llm_factory
from ragas.prompt.pydantic_prompt import PydanticPrompt


class LLMResponseModel(BaseModel):
    response: str


class _MockAsyncCreate:
    def __init__(self, response: BaseModel):
        self.response = response

    async def __call__(self, *args: t.Any, **kwargs: t.Any) -> BaseModel:
        return self.response


class _MockInstructor:
    """Instructor-like wrapper with controllable sync/async create functions.

    Class names match real Instructor wrappers so ``_check_client_async`` uses
    the right branch.
    """

    def __init__(self, response: BaseModel, is_async: bool = False):
        self.client = Mock()
        self.client.is_async = is_async
        self.chat = Mock()
        self.chat.completions = Mock()
        if is_async:
            self.chat.completions.create = _MockAsyncCreate(response)
        else:
            self.chat.completions.create = Mock(return_value=response)
        self.messages = Mock()
        self.messages.create = self.chat.completions.create

    # Class name is consumed by _check_client_async; for async wrappers use
    # "AsyncInstructor" so the heuristic returns True even when our underlying
    # mock client.chat.completions.create is the mock instance, not a real
    # coroutine function.
    class_name: str = "Instructor"

    @property
    def __class__(self):  # type: ignore[override]
        cls = type(self)
        cls.__name__ = self.class_name
        return cls


class _MockAsyncInstructor(_MockInstructor):
    class_name = "AsyncInstructor"


class _RecordingHandler(BaseCallbackHandler):
    """Captures on_llm_start / on_llm_end invocations for assertions."""

    def __init__(self) -> None:
        self.starts: t.List[t.Any] = []
        self.ends: t.List[t.Any] = []

    def on_llm_start(self, serialized, prompts, **kwargs: t.Any) -> None:
        self.starts.append({"serialized": serialized, "prompts": prompts})

    def on_llm_end(self, response, **kwargs: t.Any) -> None:
        self.ends.append(response)


@pytest.fixture
def patch_instructor(monkeypatch: pytest.MonkeyPatch):
    """Patch the active instructor factory so no real provider is contacted."""

    def factory(client, mode=None):
        is_async = getattr(client, "is_async", False)
        cls = _MockAsyncInstructor if is_async else _MockInstructor
        return cls(LLMResponseModel(response="ok"), is_async=is_async)

    # Patch the dispatch site used by llm_factory. The Anthropic / Gemini
    # branches are not exercised by these tests.
    monkeypatch.setattr("ragas.llms.base.instructor.from_openai", factory)


def test_sync_instructor_llm_generate_fires_callbacks(patch_instructor) -> None:
    """InstructorLLM.generate() should drive on_llm_start/end on callbacks."""

    client = Mock()
    client.is_async = False
    llm = llm_factory("gpt-4o-mini", provider="openai", client=client)
    assert isinstance(llm, InstructorLLM)

    recorder = _RecordingHandler()
    result = llm.generate(
        prompt="hi",
        response_model=LLMResponseModel,
        callbacks=[recorder],
    )

    assert isinstance(result, LLMResponseModel)
    assert recorder.starts, "on_llm_start was not invoked"
    assert recorder.ends, "on_llm_end was not invoked"
    assert recorder.ends[0].generations, "LLMResult.generations should be populated"


@pytest.mark.asyncio
async def test_async_instructor_llm_agenerate_fires_callbacks(patch_instructor) -> None:
    """InstructorLLM.agenerate() should drive on_llm_start/end on callbacks."""

    client = Mock()
    client.is_async = True
    llm = llm_factory("gpt-4o-mini", provider="openai", client=client)
    assert isinstance(llm, InstructorLLM)

    recorder = _RecordingHandler()
    result = await llm.agenerate(
        prompt="hi",
        response_model=LLMResponseModel,
        callbacks=[recorder],
    )

    assert isinstance(result, LLMResponseModel)
    assert recorder.starts, "async on_llm_start was not invoked"
    assert recorder.ends, "async on_llm_end was not invoked"
    assert recorder.ends[0].generations, "LLMResult.generations should be populated"


def test_pydantic_prompt_propagates_callbacks_through_sync_instructor(
    patch_instructor,
) -> None:
    """PydanticPrompt -> InstructorLLM sync path should fire CostCallbackHandler."""

    client = Mock()
    client.is_async = False
    llm = llm_factory("gpt-4o-mini", provider="openai", client=client)

    class _Input(BaseModel):
        question: str

    class _Output(BaseModel):
        response: str

    class _TestPrompt(PydanticPrompt[_Input, _Output]):
        instruction: str = "Answer the question."
        output_model: t.Type[_Output] = _Output

    prompt = _TestPrompt()
    cost_cb = CostCallbackHandler(
        token_usage_parser=lambda llm_result: TokenUsage(
            input_tokens=3, output_tokens=5, model="gpt-4o-mini"
        )
    )

    outputs = asyncio.run(
        prompt.generate(llm=llm, data=_Input(question="Q"), callbacks=[cost_cb])
    )

    assert isinstance(outputs, _Output)
    assert cost_cb.usage_data, (
        "CostCallbackHandler.usage_data should populate when callbacks run"
        " through the Instructor LLM branch."
    )


@pytest.mark.asyncio
async def test_pydantic_prompt_propagates_callbacks_through_async_instructor(
    patch_instructor,
) -> None:
    """PydanticPrompt -> InstructorLLM async path should fire callbacks."""

    client = Mock()
    client.is_async = True
    llm = llm_factory("gpt-4o-mini", provider="openai", client=client)
    assert isinstance(llm, InstructorBaseRagasLLM)

    class _Input(BaseModel):
        question: str

    class _Output(BaseModel):
        response: str

    class _TestPrompt(PydanticPrompt[_Input, _Output]):
        instruction: str = "Answer the question."
        output_model: t.Type[_Output] = _Output

    prompt = _TestPrompt()
    recorder = _RecordingHandler()

    outputs = await prompt.generate(
        llm=llm, data=_Input(question="Q"), callbacks=[recorder]
    )

    assert isinstance(outputs, _Output)
    assert recorder.starts, "async PydanticPrompt path dropped on_llm_start"
    assert recorder.ends, "async PydanticPrompt path dropped on_llm_end"
