import types
import os
import unittest.mock

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice

from ragas.llms.base import llm_factory
from ragas.llms.json_load import json_loader


class MockOpenAI:
    class MockCompletions:
        def create(self, *args, **kwargs):
            return ChatCompletion(
                id="123456",
                model="model",
                object="chat.completion",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            content='{"one": "two"},,, {"three": "four"}]',
                            role="assistant",
                        ),
                    )
                ],
                created=123,
            )

    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace()
        self.chat.completions = self.MockCompletions()


class TestJsonLoader(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.mock_environment = {
            "OPENAI_API_KEY": "abcdefg123456v",
        }
        self.patches = [
            unittest.mock.patch.dict(os.environ, self.mock_environment),
            unittest.mock.patch(
                "openai.OpenAI",
                new=MockOpenAI,
            ),
        ]

        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_json_load(self):
        broken_json = 'A broken json is here: [{"one": "two"}, {"three": "four"}'
        result = json_loader._safe_load(broken_json, llm=llm_factory())
        self.assertEqual(result, [{"one": "two"}, {"three": "four"}])
