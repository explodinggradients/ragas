import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ragas.cost import (
    CostCallbackHandler,
    TokenUsage,
    get_token_usage_for_anthropic,
    get_token_usage_for_openai,
)

"""
TODO: things to test
- get usage from LLM Result
- estimate cost works for different API providers 
- openai with multiple n
- anthropic
- anthropic with multiple n
"""


def test_token_usage():
    x = TokenUsage(input_tokens=10, output_tokens=20)
    y = TokenUsage(input_tokens=5, output_tokens=15)
    assert (x + y).input_tokens == 15
    assert (x + y).output_tokens == 35

    with pytest.raises(ValueError):
        x.model = "openai"
        y.model = "gpt3"
        _ = x + y

    # test equals
    assert x == x
    assert y != x
    z = TokenUsage(input_tokens=10, output_tokens=20)
    z_with_model = TokenUsage(input_tokens=10, output_tokens=20, model="openai")
    z_same_with_model = TokenUsage(input_tokens=10, output_tokens=20, model="openai")
    assert z_with_model != z
    assert z_same_with_model == z_with_model

    # test same model
    assert z_with_model.is_same_model(z_same_with_model)
    assert not z_with_model.is_same_model(z)


def test_token_usage_cost():
    x = TokenUsage(input_tokens=10, output_tokens=20)
    assert x.cost(cost_per_input_token=0.1, cost_per_output_token=0.2) == 5.0


openai_llm_result = LLMResult(
    generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
    llm_output={
        "token_usage": {
            "completion_tokens": 10,
            "prompt_tokens": 10,
            "total_tokens": 20,
        },
        "model_name": "gpt-4o",
        "system_fingerprint": "fp_2eie",
    },
)

athropic_llm_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                message=AIMessage(
                    content="Hello, world!",
                    response_metadata={
                        "id": "msg_01UHjFfUr",
                        "model": "claude-3-opus-20240229",
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 9, "output_tokens": 12},
                    },
                )
            )
        ]
    ],
    llm_output={},
)


def test_parse_llm_results():
    # openai
    token_usage = get_token_usage_for_openai(openai_llm_result)
    assert token_usage == TokenUsage(input_tokens=10, output_tokens=10)

    # anthropic
    token_usage = get_token_usage_for_anthropic(athropic_llm_result)
    assert token_usage == TokenUsage(input_tokens=9, output_tokens=12)


def test_cost_callback_handler():
    cost_cb = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    cost_cb.on_llm_end(openai_llm_result)

    # cost
    assert cost_cb.total_tokens() == TokenUsage(input_tokens=10, output_tokens=10)

    assert cost_cb.total_cost(0.1) == 2.0
    assert (
        cost_cb.total_cost(cost_per_input_token=0.1, cost_per_output_token=0.1) == 2.0
    )
