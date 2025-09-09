import os

import pytest

from ragas.llms import llm_factory
from ragas.metrics import context_recall


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_adapt():
    llm = llm_factory("gpt-4o")
    await context_recall.adapt_prompts(llm=llm, language="spanish")
    assert context_recall.context_recall_prompt.language == "spanish"
