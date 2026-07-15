import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ragas.cost import (
    CostCallbackHandler,
    TokenUsage,
    get_token_usage_for_anthropic,
    get_token_usage_for_azure_ai,
    get_token_usage_for_bedrock,
    get_token_usage_for_cohere,
    get_token_usage_for_google_genai,
    get_token_usage_for_langchain_chat_models,
    get_token_usage_for_mistral,
    get_token_usage_for_openai,
    get_token_usage_for_vertex_ai,
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

anthropic_llm_result = LLMResult(
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

bedrock_llama_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                text="Hello, world!",
                message=AIMessage(
                    content="Hello, world!",
                    response_metadata={
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 10,
                            "total_tokens": 20,
                        },
                        "stop_reason": "stop",
                        "model_id": "us.meta.llama3-1-70b-instruct-v1:0",
                    },
                ),
            )
        ]
    ],
    llm_output={},
)

bedrock_claude_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                text="Hello, world!",
                message=AIMessage(
                    content="Hello, world!",
                    response_metadata={
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 10,
                            "total_tokens": 20,
                        },
                        "stop_reason": "end_turn",
                        "model_id": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                    },
                ),
            )
        ]
    ],
    llm_output={},
)

azure_ai_result = LLMResult(
    generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
    llm_output={
        "token_usage": {
            "input_tokens": 10,
            "output_tokens": 10,
            "total_tokens": 20,
        },
        "model_name": "mistral-small-2503",
    },
)

# Real langchain-google-genai shape: usage_metadata on message, model_name in
# response_metadata.
google_genai_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                message=AIMessage(
                    content="Hello, world!",
                    usage_metadata={
                        "input_tokens": 12,
                        "output_tokens": 826,
                        "total_tokens": 838,
                        "input_token_details": {"cache_read": 0},
                        "output_token_details": {"reasoning": 777},
                    },
                    response_metadata={
                        "model_provider": "google_genai",
                        "model_name": "models/gemini-2.5-flash",
                    },
                )
            )
        ]
    ],
    llm_output={},
)

# Real langchain-google-vertexai shape.
vertex_ai_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                message=AIMessage(
                    content="Hello, world!",
                    usage_metadata={
                        "input_tokens": 18,
                        "output_tokens": 1,
                        "total_tokens": 19,
                    },
                    response_metadata={
                        "model_provider": "google_vertexai",
                        "model_name": "gemini-1.5-pro",
                    },
                )
            )
        ]
    ],
    llm_output={},
)

# Real langchain-mistralai shape (stream-final chunk sets model_name).
mistral_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                message=AIMessage(
                    content="Hello, world!",
                    usage_metadata={
                        "input_tokens": 32,
                        "output_tokens": 52,
                        "total_tokens": 84,
                    },
                    response_metadata={
                        "model_provider": "mistralai",
                        "model_name": "mistral-small",
                        "finish_reason": "stop",
                    },
                )
            )
        ]
    ],
    llm_output={},
)

# Real langchain-cohere shape.
cohere_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                message=AIMessage(
                    content="Hello, world!",
                    usage_metadata={
                        "input_tokens": 24,
                        "output_tokens": 7,
                        "total_tokens": 31,
                    },
                    response_metadata={
                        "model_provider": "cohere",
                        "model_name": "command-r-plus",
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
    assert token_usage == TokenUsage(input_tokens=10, output_tokens=10, model="gpt-4o")

    # anthropic
    token_usage = get_token_usage_for_anthropic(anthropic_llm_result)
    assert token_usage == TokenUsage(
        input_tokens=9, output_tokens=12, model="claude-3-opus-20240229"
    )

    # Bedrock LLaMa
    token_usage = get_token_usage_for_bedrock(bedrock_llama_result)
    assert token_usage == TokenUsage(
        input_tokens=10, output_tokens=10, model="us.meta.llama3-1-70b-instruct-v1:0"
    )

    # Bedrock Claude
    token_usage = get_token_usage_for_bedrock(bedrock_claude_result)
    assert token_usage == TokenUsage(
        input_tokens=10,
        output_tokens=10,
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    )

    # Azure AI
    token_usage = get_token_usage_for_azure_ai(azure_ai_result)
    assert token_usage == TokenUsage(
        input_tokens=10, output_tokens=10, model="mistral-small-2503"
    )


def test_azure_ai_edge_cases():
    # Test with None llm_output
    empty_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
        llm_output=None,
    )
    token_usage = get_token_usage_for_azure_ai(empty_result)
    assert token_usage == TokenUsage(input_tokens=0, output_tokens=0)

    # Test with empty llm_output
    empty_llm_output_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
        llm_output={},
    )
    token_usage = get_token_usage_for_azure_ai(empty_llm_output_result)
    assert token_usage == TokenUsage(input_tokens=0, output_tokens=0)

    # Test with missing token_usage field
    no_token_usage_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
        llm_output={"model_name": "mistral-small-2503"},
    )
    token_usage = get_token_usage_for_azure_ai(no_token_usage_result)
    assert token_usage == TokenUsage(
        input_tokens=0, output_tokens=0, model="mistral-small-2503"
    )

    # Test with partial token_usage field
    partial_token_usage_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
        llm_output={
            "token_usage": {"input_tokens": 15},  # missing output_tokens
            "model_name": "mistral-small-2503",
        },
    )
    token_usage = get_token_usage_for_azure_ai(partial_token_usage_result)
    assert token_usage == TokenUsage(
        input_tokens=15, output_tokens=0, model="mistral-small-2503"
    )


def test_cost_callback_handler():
    cost_cb = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    cost_cb.on_llm_end(openai_llm_result)

    # cost
    assert cost_cb.total_tokens() == TokenUsage(
        input_tokens=10, output_tokens=10, model="gpt-4o"
    )

    assert cost_cb.total_cost(0.1) == 2.0
    assert (
        cost_cb.total_cost(cost_per_input_token=0.1, cost_per_output_token=0.1) == 2.0
    )


def test_parse_llm_results_modern_chat_providers():
    # Google GenAI
    token_usage = get_token_usage_for_google_genai(google_genai_result)
    assert token_usage == TokenUsage(
        input_tokens=12, output_tokens=826, model="models/gemini-2.5-flash"
    )

    # Vertex AI
    token_usage = get_token_usage_for_vertex_ai(vertex_ai_result)
    assert token_usage == TokenUsage(
        input_tokens=18, output_tokens=1, model="gemini-1.5-pro"
    )

    # Mistral
    token_usage = get_token_usage_for_mistral(mistral_result)
    assert token_usage == TokenUsage(
        input_tokens=32, output_tokens=52, model="mistral-small"
    )

    # Cohere
    token_usage = get_token_usage_for_cohere(cohere_result)
    assert token_usage == TokenUsage(
        input_tokens=24, output_tokens=7, model="command-r-plus"
    )


def test_langchain_chat_models_parser_edge_cases():
    # Empty generations → zeros
    empty_result = LLMResult(generations=[], llm_output={})
    assert get_token_usage_for_langchain_chat_models(empty_result) == TokenUsage(
        input_tokens=0, output_tokens=0
    )

    # ChatGeneration without usage_metadata → zeros, no model
    no_usage = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="hi",
                        response_metadata={"model_name": "anything"},
                    )
                )
            ]
        ],
        llm_output={},
    )
    assert get_token_usage_for_langchain_chat_models(no_usage) == TokenUsage(
        input_tokens=0, output_tokens=0, model="anything"
    )

    # usage_metadata present, no response_metadata → counts, no model
    no_model = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="hi",
                        usage_metadata={
                            "input_tokens": 5,
                            "output_tokens": 7,
                            "total_tokens": 12,
                        },
                    )
                )
            ]
        ],
        llm_output={},
    )
    assert get_token_usage_for_langchain_chat_models(no_model) == TokenUsage(
        input_tokens=5, output_tokens=7, model=""
    )

    # Partial usage_metadata (only input_tokens)
    partial = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="hi",
                        usage_metadata={
                            "input_tokens": 11,
                            "output_tokens": 0,
                            "total_tokens": 11,
                        },
                        response_metadata={"model_name": "any-model"},
                    )
                )
            ]
        ],
        llm_output={},
    )
    assert get_token_usage_for_langchain_chat_models(partial) == TokenUsage(
        input_tokens=11, output_tokens=0, model="any-model"
    )

    # usage_metadata explicitly None
    explicit_none = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="hi",
                        usage_metadata=None,
                        response_metadata={"model_name": "any-model"},
                    )
                )
            ]
        ],
        llm_output={},
    )
    assert get_token_usage_for_langchain_chat_models(explicit_none) == TokenUsage(
        input_tokens=0, output_tokens=0, model="any-model"
    )

    # Multi-generation sum (n > 1) — picks first non-empty model
    multi = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="a",
                        usage_metadata={
                            "input_tokens": 2,
                            "output_tokens": 3,
                            "total_tokens": 5,
                        },
                        response_metadata={"model_name": "models/multi"},
                    )
                ),
                ChatGeneration(
                    message=AIMessage(
                        content="b",
                        usage_metadata={
                            "input_tokens": 4,
                            "output_tokens": 5,
                            "total_tokens": 9,
                        },
                        response_metadata={"model_name": "models/multi"},
                    )
                ),
            ]
        ],
        llm_output={},
    )
    assert get_token_usage_for_langchain_chat_models(multi) == TokenUsage(
        input_tokens=6, output_tokens=8, model="models/multi"
    )


def test_langchain_chat_models_parser_handles_non_dict_usage_metadata():
    # Some downstream code may set `usage_metadata` to a non-dict after
    # construction. The parser must fall back to zeros rather than raise.
    weird_msg = AIMessage(content="hi", response_metadata={"model_name": "any-model"})
    # Bypass pydantic validation to inject a non-dict value (the realistic
    # # case is downstream code reassigning the attribute).
    object.__setattr__(weird_msg, "usage_metadata", "not-a-dict")
    weird = LLMResult(generations=[[ChatGeneration(message=weird_msg)]], llm_output={})
    assert get_token_usage_for_langchain_chat_models(weird) == TokenUsage(
        input_tokens=0, output_tokens=0, model="any-model"
    )
