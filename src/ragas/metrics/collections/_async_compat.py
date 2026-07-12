"""Async-compatible dispatch helpers for collections metrics.

The collections metrics call ``await self.llm.agenerate(prompt, response_model)``
against an ``InstructorLLM`` that may be either async- or sync-capable depending
on the underlying client (sync ``OpenAI``, sync ``Anthropic``, ``google-genai``
``Client`` are all classified as sync). When ``InstructorLLM.is_async`` is
``False``, calling ``agenerate`` raises a strict ``TypeError``.

This module provides ``call_llm_async`` so collections metrics can use a single
async entry point that:

- delegates to ``await llm.agenerate(prompt, response_model)`` for async LLMs,
- runs ``llm.generate(prompt, response_model)`` in a worker thread for sync
  LLMs (so the event loop is not blocked).

Keeping the dispatch in one helper means individual metrics don't need to
duplicate the sync-vs-async branch.
"""

from __future__ import annotations

import asyncio
import typing as t

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


async def call_llm_async(
    llm: "InstructorBaseRagasLLM",
    prompt: str,
    response_model: t.Type[t.Any],
) -> t.Any:
    """Call an InstructorLLM, transparently supporting sync clients.

    Mirrors the (prompt, response_model) signature used by every collections
    metric, so callers can swap ``await self.llm.agenerate(...)`` for
    ``await call_llm_async(self.llm, ...)`` without changing the rest of the
    function.
    """
    is_async = bool(getattr(llm, "is_async", False))
    if is_async:
        return await llm.agenerate(prompt=prompt, response_model=response_model)
    # InstructorLLM.generate is sync; offload it so the running event loop
    # stays responsive when collections.ascore() is awaited.
    return await asyncio.to_thread(
        llm.generate, prompt=prompt, response_model=response_model
    )
