"""Regression tests for LangchainLLMWrapper.is_finished finish-reason parsing.

Covers https://github.com/vibrantlabsai/ragas/issues/1548 — providers report
different finish_reason values (e.g. TogetherAI-hosted Llama uses "eos").
"""

from __future__ import annotations

import warnings

from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from ragas.llms.base import LangchainLLMWrapper


def _wrapper() -> LangchainLLMWrapper:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(FakeListLLM(responses=["ok"]))


def _from_generation_info(finish_reason: str) -> LLMResult:
    gen = Generation(text="ok", generation_info={"finish_reason": finish_reason})
    return LLMResult(generations=[[gen]])


def _from_response_metadata(finish_reason: str) -> LLMResult:
    msg = AIMessage(content="ok", response_metadata={"finish_reason": finish_reason})
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_is_finished_accepts_eos_from_generation_info():
    # TogetherAI-hosted Llama reports finish_reason="eos" (issue #1548).
    assert _wrapper().is_finished(_from_generation_info("eos")) is True


def test_is_finished_accepts_eos_from_response_metadata():
    assert _wrapper().is_finished(_from_response_metadata("eos")) is True


def test_is_finished_accepts_existing_reasons():
    wrapper = _wrapper()
    assert wrapper.is_finished(_from_generation_info("stop")) is True
    assert wrapper.is_finished(_from_generation_info("eos_token")) is True


def test_is_finished_false_when_truncated():
    # "length" means the output was cut off — not a normal finish.
    assert _wrapper().is_finished(_from_generation_info("length")) is False
