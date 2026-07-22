"""Regression tests for exception narrowing in PydanticPrompt.generate_multiple.

Before the fix, ``except RagasOutputParserException`` only caught that one type.
The LangChain / Instructor path raises ``pydantic.ValidationError`` from
``model_validate_json``, and the RagasLLM retry fallback can raise
``langchain_core.exceptions.OutputParserException``. Both bypassed
``on_chain_error`` and leaked the raw exception type to callers.
"""

from __future__ import annotations

import typing as t
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import Generation, LLMResult
from pydantic import BaseModel, ValidationError

from ragas.exceptions import RagasOutputParserException
from ragas.llms.base import BaseRagasLLM
from ragas.prompt import StringIO
from ragas.prompt.multi_modal_prompt import ImageTextPrompt
from ragas.prompt.pydantic_prompt import PydanticPrompt, RagasOutputParser
from ragas.run_config import RunConfig


class _Out(BaseModel):
    foo: int


class _Prompt(PydanticPrompt[StringIO, _Out]):
    instruction = "Return JSON with an integer foo."
    input_model = StringIO
    output_model = _Out


class _ImageIn(BaseModel):
    text: str

    def to_string_list(self) -> t.List[str]:
        return [self.text]


class _ImagePrompt(ImageTextPrompt[_ImageIn, _Out]):
    instruction = "Return JSON with an integer foo."
    input_model = _ImageIn
    output_model = _Out


class _EchoLLM(BaseRagasLLM):
    def __init__(self, text: str = '{"foo": 1}'):
        super().__init__(run_config=RunConfig())
        self._text = text

    def generate_text(self, prompt, *args, **kwargs) -> LLMResult:  # type: ignore[no-untyped-def]
        return LLMResult(generations=[[Generation(text=self._text)]])

    async def agenerate_text(self, prompt, *args, **kwargs) -> LLMResult:  # type: ignore[no-untyped-def]
        return LLMResult(generations=[[Generation(text=self._text)]])

    def is_finished(self, response: LLMResult) -> bool:
        return True


class _RecordingHandler(BaseCallbackHandler):
    """Captures on_chain_error / on_chain_end for assertions."""

    def __init__(self) -> None:
        self.errors: t.List[BaseException] = []
        self.ends: t.List[t.Dict[str, t.Any]] = []

    def on_chain_error(self, error: BaseException, **kwargs: t.Any) -> None:
        self.errors.append(error)

    def on_chain_end(self, outputs: t.Dict[str, t.Any], **kwargs: t.Any) -> None:
        self.ends.append(outputs)


def _langchain_llm_returning(text: str) -> MagicMock:
    llm = MagicMock()
    llm.agenerate_prompt = AsyncMock(
        return_value=LLMResult(generations=[[Generation(text=text)]])
    )
    return llm


@pytest.mark.asyncio
async def test_langchain_path_validation_error_fires_chain_error_callback():
    handler = _RecordingHandler()
    llm = _langchain_llm_returning('{"foo": "not_an_int"}')

    with patch("ragas.prompt.pydantic_prompt.is_langchain_llm", return_value=True):
        with pytest.raises(RagasOutputParserException) as exc_info:
            await _Prompt().generate_multiple(
                llm=llm,
                data=StringIO(text="hi"),
                callbacks=[handler],
            )

    assert isinstance(exc_info.value.__cause__, ValidationError)
    assert len(handler.errors) >= 1
    assert isinstance(handler.errors[-1], ValidationError)
    assert handler.ends == []


@pytest.mark.asyncio
async def test_ragas_llm_retry_fallback_output_parser_exception_fires_chain_error():
    handler = _RecordingHandler()
    llm = _EchoLLM(text="not-json-at-all")

    async def _raise_output_parser_exception(*args: t.Any, **kwargs: t.Any) -> t.Any:
        raise OutputParserException("forced parse failure")

    with patch.object(
        RagasOutputParser,
        "parse_output_string",
        side_effect=_raise_output_parser_exception,
    ):
        with pytest.raises(RagasOutputParserException) as exc_info:
            await _Prompt().generate_multiple(
                llm=llm,
                data=StringIO(text="hi"),
                callbacks=[handler],
            )

    assert isinstance(exc_info.value.__cause__, OutputParserException)
    assert len(handler.errors) >= 1
    assert isinstance(handler.errors[-1], OutputParserException)
    assert handler.ends == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_mode",
    ["validation_error", "output_parser_exception", "ragas_parser_exception"],
)
async def test_chain_end_not_called_on_any_failure_path(failure_mode: str):
    handler = _RecordingHandler()

    if failure_mode == "validation_error":
        llm = _langchain_llm_returning('{"foo": "not_an_int"}')
        with patch("ragas.prompt.pydantic_prompt.is_langchain_llm", return_value=True):
            with pytest.raises(RagasOutputParserException):
                await _Prompt().generate_multiple(
                    llm=llm,
                    data=StringIO(text="hi"),
                    callbacks=[handler],
                )
    elif failure_mode == "output_parser_exception":
        llm = _EchoLLM()

        async def _raise_ope(*args: t.Any, **kwargs: t.Any) -> t.Any:
            raise OutputParserException("forced")

        with patch.object(
            RagasOutputParser, "parse_output_string", side_effect=_raise_ope
        ):
            with pytest.raises(RagasOutputParserException):
                await _Prompt().generate_multiple(
                    llm=llm,
                    data=StringIO(text="hi"),
                    callbacks=[handler],
                )
    else:
        llm = _EchoLLM()

        async def _raise_ragas(*args: t.Any, **kwargs: t.Any) -> t.Any:
            raise RagasOutputParserException()

        with patch.object(
            RagasOutputParser, "parse_output_string", side_effect=_raise_ragas
        ):
            with pytest.raises(RagasOutputParserException):
                await _Prompt().generate_multiple(
                    llm=llm,
                    data=StringIO(text="hi"),
                    callbacks=[handler],
                )

    assert handler.ends == []
    assert len(handler.errors) >= 1


@pytest.mark.asyncio
async def test_image_text_prompt_langchain_path_validation_error_fires_chain_error():
    handler = _RecordingHandler()
    llm = _langchain_llm_returning('{"foo": "not_an_int"}')

    with (
        patch("ragas.prompt.multi_modal_prompt.is_langchain_llm", return_value=True),
        patch("ragas.prompt.pydantic_prompt.is_langchain_llm", return_value=True),
    ):
        with pytest.raises(RagasOutputParserException) as exc_info:
            await _ImagePrompt().generate_multiple(
                llm=llm,
                data=_ImageIn(text="hi"),
                callbacks=[handler],
            )

    assert isinstance(exc_info.value.__cause__, ValidationError)
    assert len(handler.errors) >= 1
    assert isinstance(handler.errors[-1], ValidationError)
    assert handler.ends == []
