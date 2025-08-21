from ragas.llms import llm_factory
from ragas.metrics import context_recall


async def test_adapt():
    llm = llm_factory("gpt-4o")
    await context_recall.adapt_prompts(llm=llm, language="spanish")
    assert context_recall.context_recall_prompt.language == "spanish"
