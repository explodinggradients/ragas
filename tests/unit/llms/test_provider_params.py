"""Tests for InstructorLLM provider-specific parameter mapping.

Covers the Anthropic top_p deprecation fix (#2674) and regression coverage
for the existing Google and OpenAI/Azure mapping branches.
"""

from unittest.mock import Mock

from ragas.llms.base import InstructorLLM


def _make_llm(provider: str, model: str = "test-model", **kwargs) -> InstructorLLM:
    """Build an InstructorLLM with a minimal mock client for param-mapping tests."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock()
    return InstructorLLM(client=client, model=model, provider=provider, **kwargs)


class TestAnthropicParamMapping:
    """Regression tests for issue #2674: deprecated top_p sent to Anthropic."""

    def test_default_top_p_is_stripped(self):
        """Default top_p=0.1 must not be forwarded to Anthropic (would 400)."""
        llm = _make_llm("anthropic", model="claude-sonnet-4-6")

        mapped = llm._map_provider_params()

        assert "top_p" not in mapped
        assert mapped["temperature"] == 0.01
        assert mapped["max_tokens"] == 1024

    def test_explicit_top_p_is_stripped(self):
        """Even an explicit top_p is stripped because temperature is always
        present and Anthropic rejects sending both on current-gen models."""
        llm = _make_llm("anthropic", model="claude-sonnet-4-6", top_p=0.5)

        mapped = llm._map_provider_params()

        assert "top_p" not in mapped

    def test_other_params_preserved(self):
        """temperature and max_tokens must survive the Anthropic mapping."""
        llm = _make_llm(
            "anthropic",
            model="claude-sonnet-4-6",
            temperature=0.0,
            max_tokens=2048,
        )

        mapped = llm._map_provider_params()

        assert mapped["temperature"] == 0.0
        assert mapped["max_tokens"] == 2048
        assert "top_p" not in mapped

    def test_anthropic_branch_invoked(self):
        """provider='anthropic' must route to _map_anthropic_params, not pass-through."""
        llm = _make_llm("anthropic", model="claude-sonnet-4-6")

        mapped = llm._map_provider_params()

        # If pass-through had been used, top_p=0.1 would still be present.
        assert "top_p" not in mapped

    def test_anthropic_case_insensitive(self):
        """Provider matching should be case-insensitive (uses .lower())."""
        llm = _make_llm("Anthropic", model="claude-sonnet-4-6")

        mapped = llm._map_provider_params()

        assert "top_p" not in mapped

    def test_does_not_mutate_model_args(self):
        """_map_provider_params must not mutate self.model_args (returns a copy)."""
        llm = _make_llm("anthropic", model="claude-sonnet-4-6")
        original = llm.model_args.copy()

        _ = llm._map_provider_params()

        assert llm.model_args == original
        assert "top_p" in llm.model_args  # still present on the instance


class TestNonAnthropicRegression:
    """Ensure the Anthropic branch does not affect other providers."""

    def test_google_still_wraps_generation_config(self):
        llm = _make_llm("google", model="gemini-2.0-flash")

        mapped = llm._map_provider_params()

        # Google wraps temperature/max_tokens/top_p inside generation_config.
        assert "generation_config" in mapped
        gc = mapped["generation_config"]
        assert gc["temperature"] == 0.01
        assert gc["max_output_tokens"] == 1024
        assert gc["top_p"] == 0.1

    def test_openai_legacy_passes_top_p(self):
        """Legacy OpenAI models (gpt-4) should still receive top_p unchanged."""
        llm = _make_llm("openai", model="gpt-4o")

        mapped = llm._map_provider_params()

        # gpt-4o is not a reasoning model, so top_p is preserved.
        assert mapped["top_p"] == 0.1
        assert mapped["max_tokens"] == 1024

    def test_openai_reasoning_strips_top_p(self):
        """Reasoning models (o-series, gpt-5+) strip top_p via _map_openai_params."""
        llm = _make_llm("openai", model="o3-mini")

        mapped = llm._map_provider_params()

        assert "top_p" not in mapped
        assert mapped["temperature"] == 1.0
        assert mapped["max_completion_tokens"] == 1024

    def test_litellm_passes_through(self):
        """LiteLLM is the fallback pass-through; top_p must remain."""
        llm = _make_llm("litellm", model="gpt-4o")

        mapped = llm._map_provider_params()

        assert mapped["top_p"] == 0.1
        assert mapped["temperature"] == 0.01
