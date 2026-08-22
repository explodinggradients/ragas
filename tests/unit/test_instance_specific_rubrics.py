import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

from ragas.llms.base import BaseRagasLLM
from ragas.metrics._instance_specific_rubrics import InstanceRubrics
from ragas.run_config import RunConfig


class _CapturingLLM(BaseRagasLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_prompts = []

    def generate_text(self, prompt: PromptValue, *args, **kwargs) -> LLMResult:
        self.captured_prompts.append(prompt.to_string())
        return LLMResult(
            generations=[[Generation(text='{"score": 5, "feedback": "ok"}')]]
        )

    async def agenerate_text(self, prompt: PromptValue, *args, **kwargs) -> LLMResult:
        return self.generate_text(prompt, *args, **kwargs)

    def is_finished(self, response: LLMResult) -> bool:
        return True


@pytest.mark.asyncio
async def test_user_input_none_with_contexts_does_not_emit_none():
    llm = _CapturingLLM(run_config=RunConfig())
    metric = InstanceRubrics(llm=llm)
    await metric._ascore(
        row={
            "user_input": None,
            "retrieved_contexts": ["ctx-1", "ctx-2"],
            "response": "some response",
            "reference": None,
            "rubrics": {"score1_description": "bad", "score5_description": "great"},
        },
        callbacks=[],
    )
    assert "None answer using context" not in llm.captured_prompts[0]


@pytest.mark.asyncio
async def test_user_input_string_with_contexts_unchanged():
    llm = _CapturingLLM(run_config=RunConfig())
    metric = InstanceRubrics(llm=llm)
    await metric._ascore(
        row={
            "user_input": "What is X?",
            "retrieved_contexts": ["ctx-1"],
            "response": "some response",
            "reference": None,
            "rubrics": {"score1_description": "bad", "score5_description": "great"},
        },
        callbacks=[],
    )
    assert "What is X? answer using context: ctx-1" in llm.captured_prompts[0]
