"""Regression tests for ragas.optimizers.dspy_llm_wrapper.RagasDSPyLM.

Covers the `inspect_history` Python ``[-0:]`` slice quirk. Before the
fix, ``inspect_history(n=0)`` returned the entire history because
``[-0:]`` is equivalent to ``[0:]``.

The tests populate ``wrapper.history`` directly so they do not depend on
the async ``_generate`` path (which has its own unrelated import issue
tracked separately).
"""

from __future__ import annotations

from ragas.llms.base import BaseRagasLLM
from ragas.optimizers.dspy_llm_wrapper import RagasDSPyLM
from ragas.run_config import RunConfig


class _StubLLM(BaseRagasLLM):
    """Minimal LLM stub. Not exercised by these tests; present only so the
    RagasDSPyLM constructor accepts its dependency."""

    def generate_text(self, prompt, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def agenerate_text(self, prompt, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def is_finished(self, response) -> bool:  # type: ignore[override]
        return True


def _make_wrapper() -> RagasDSPyLM:
    return RagasDSPyLM(ragas_llm=_StubLLM(run_config=RunConfig()))


def test_inspect_history_n_zero_returns_empty() -> None:
    wrapper = _make_wrapper()
    wrapper.history = [{"call": 1}, {"call": 2}, {"call": 3}]

    assert wrapper.inspect_history(n=0) == []


def test_inspect_history_n_negative_returns_empty() -> None:
    wrapper = _make_wrapper()
    wrapper.history = [{"call": 1}, {"call": 2}, {"call": 3}]

    assert wrapper.inspect_history(n=-1) == []
    assert wrapper.inspect_history(n=-5) == []


def test_inspect_history_n_one_returns_last_entry() -> None:
    wrapper = _make_wrapper()
    wrapper.history = [{"call": 1}, {"call": 2}, {"call": 3}]

    last = wrapper.inspect_history(n=1)
    assert len(last) == 1
    assert last[0] == {"call": 3}


def test_inspect_history_n_at_history_length_returns_full_history() -> None:
    wrapper = _make_wrapper()
    history = [{"call": 1}, {"call": 2}, {"call": 3}]
    wrapper.history = history

    assert wrapper.inspect_history(n=3) == history


def test_inspect_history_n_larger_than_history_returns_full_history() -> None:
    wrapper = _make_wrapper()
    history = [{"call": 1}, {"call": 2}]
    wrapper.history = history

    assert wrapper.inspect_history(n=10) == history


def test_inspect_history_on_empty_history_is_empty_for_any_n() -> None:
    wrapper = _make_wrapper()
    wrapper.history = []

    assert wrapper.inspect_history(n=0) == []
    assert wrapper.inspect_history(n=1) == []
    assert wrapper.inspect_history(n=10) == []
