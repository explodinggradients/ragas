import os
import unittest
import unittest.mock
from typing import Any, List, Optional
import typing as t

from langchain_core.callbacks import Callbacks, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

from ragas.llms.base import llm_factory, BaseRagasLLM
from ragas.llms.json_load import json_loader
from ragas.llms.prompt import PromptValue


class MockRagasLLM(BaseRagasLLM):
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        return LLMResult(
            generations=[
                ChatGeneration(
                    message=AIMessage('{"one": "two"},,, {"three": "four"}]')
                )
            ]
        )


class TestJsonLoader(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.mock_environment = {
            "OPENAI_API_KEY": "abcdefg123456v",
        }

        self.patches = [
            unittest.mock.patch.dict(os.environ, self.mock_environment),
            unittest.mock.patch(
                "ragas.llms.LangchainLLMWrapper",
                new=MockRagasLLM,
            ),
        ]

        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_json_load(self):
        broken_json = 'A broken json is here: [{"one": "two"}, {"three": "four"}'
        result = json_loader._safe_load(broken_json, llm=llm_factory())
        self.assertEqual(result, [{"one": "two"}, {"three": "four"}])
