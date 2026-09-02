import pytest

from ragas.llms.base import InstructorLLM

REASONING_MODEL_EXPECTED = {
    # GPT-5.x
    "gpt-5.6-sol": True,
    "gpt-5.6-terra": True,
    "gpt-5.6-luna": True,
    "gpt-5.5": True,
    "gpt-5.4": True,
    "gpt-5.4-mini": True,
    "gpt-5.4-nano": True,
    "gpt-5.4-mini-2026-03-17": True,
    "gpt-5.4-nano-2026-03-17": True,
    "gpt-5.3-chat-latest": True,
    "gpt-5.2": True,
    "gpt-5.2-2025-12-11": True,
    "gpt-5.2-chat-latest": True,
    "gpt-5.2-pro": True,
    "gpt-5.2-pro-2025-12-11": True,
    "gpt-5.1": True,
    "gpt-5.1-2025-11-13": True,
    "gpt-5.1-codex": True,
    "gpt-5.1-mini": True,
    "gpt-5.1-chat-latest": True,
    "gpt-5": True,
    "gpt-5-mini": True,
    "gpt-5-nano": True,
    "gpt-5-2025-08-07": True,
    "gpt-5-mini-2025-08-07": True,
    "gpt-5-nano-2025-08-07": True,
    "gpt-5-chat-latest": True,
    # GPT-4.1
    "gpt-4.1": False,
    "gpt-4.1-mini": False,
    "gpt-4.1-nano": False,
    "gpt-4.1-2025-04-14": False,
    "gpt-4.1-mini-2025-04-14": False,
    "gpt-4.1-nano-2025-04-14": False,
    # o-series
    "o4-mini": True,
    "o4-mini-2025-04-16": True,
    "o3": True,
    "o3-2025-04-16": True,
    "o3-mini": True,
    "o3-mini-2025-01-31": True,
    "o1": True,
    "o1-2024-12-17": True,
    "o1-preview": True,
    "o1-preview-2024-09-12": True,
    "o1-mini": True,
    "o1-mini-2024-09-12": True,
    # GPT-4o
    "gpt-4o": False,
    "gpt-4o-2024-11-20": False,
    "gpt-4o-2024-08-06": False,
    "gpt-4o-2024-05-13": False,
    "gpt-4o-audio-preview": False,
    "gpt-4o-audio-preview-2024-10-01": False,
    "gpt-4o-audio-preview-2024-12-17": False,
    "gpt-4o-audio-preview-2025-06-03": False,
    "gpt-4o-mini-audio-preview": False,
    "gpt-4o-mini-audio-preview-2024-12-17": False,
    "gpt-4o-search-preview": False,
    "gpt-4o-mini-search-preview": False,
    "gpt-4o-search-preview-2025-03-11": False,
    "gpt-4o-mini-search-preview-2025-03-11": False,
    "chatgpt-4o-latest": False,
    "gpt-4o-mini": False,
    "gpt-4o-mini-2024-07-18": False,
    # Codex
    "codex-mini-latest": True,
    # GPT-4
    "gpt-4-turbo": False,
    "gpt-4-turbo-2024-04-09": False,
    "gpt-4-0125-preview": False,
    "gpt-4-turbo-preview": False,
    "gpt-4-1106-preview": False,
    "gpt-4-vision-preview": False,
    "gpt-4": False,
    "gpt-4-0314": False,
    "gpt-4-0613": False,
    "gpt-4-32k": False,
    "gpt-4-32k-0314": False,
    "gpt-4-32k-0613": False,
    # GPT-3.5
    "gpt-3.5-turbo": False,
    "gpt-3.5-turbo-16k": False,
    "gpt-3.5-turbo-0301": False,
    "gpt-3.5-turbo-0613": False,
    "gpt-3.5-turbo-1106": False,
    "gpt-3.5-turbo-0125": False,
    "gpt-3.5-turbo-16k-0613": False,
}


@pytest.mark.parametrize(
    ("model", "expected"),
    REASONING_MODEL_EXPECTED.items(),
)
def test_is_reasoning_model(model, expected):
    llm = InstructorLLM(client=None, model=model, provider="openai")
    result = llm._map_openai_params()
    if expected:
        assert "max_completion_tokens" in result
        assert "max_tokens" not in result
    else:
        assert "max_tokens" in result
        assert "max_completion_tokens" not in result
