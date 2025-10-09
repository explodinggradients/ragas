"""End-to-end tests specifically for AnswerRelevancyV2 metric functionality."""

import os

import pytest

from ragas.metrics.collections._answer_relevancy import (
    AnswerRelevancy as OriginalAnswerRelevancy,
)
from ragas.metrics.collections._answer_relevancy_v2 import AnswerRelevancy


class TestAnswerRelevancyV2E2E:
    """End-to-end tests for AnswerRelevancyV2 metric with real LLM and embeddings."""

    @pytest.fixture
    def openai_api_key(self):
        """Get OpenAI API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set - skipping AnswerRelevancyV2 E2E tests")
        return api_key

    @pytest.fixture
    def real_llm(self, openai_api_key):
        """Create real OpenAI instructor LLM."""
        try:
            import openai

            from ragas.llms.base import instructor_llm_factory

            client = openai.AsyncOpenAI(api_key=openai_api_key)
            return instructor_llm_factory("openai", client=client, model="gpt-4o-mini")
        except ImportError as e:
            pytest.skip(f"OpenAI not available: {e}")

    @pytest.fixture
    def real_embeddings(self, openai_api_key):
        """Create real OpenAI embeddings."""
        try:
            import openai

            from ragas.embeddings.base import embedding_factory

            client = openai.AsyncOpenAI(api_key=openai_api_key)
            return embedding_factory(
                provider="openai",
                model="text-embedding-ada-002",
                client=client,
                interface="modern",
            )
        except ImportError as e:
            pytest.skip(f"OpenAI embeddings not available: {e}")

    @pytest.mark.asyncio
    async def test_json_prompt_vs_string_prompt_comparison(
        self, real_llm, real_embeddings
    ):
        """Test that JSON prompt version (V2) produces similar results to string prompt version."""
        print("\n🆚 Comparing JSON prompt (V2) vs String prompt (Original)")

        # Create both metrics
        json_prompt_metric = AnswerRelevancy(
            llm=real_llm, embeddings=real_embeddings, strictness=3
        )
        string_prompt_metric = OriginalAnswerRelevancy(
            llm=real_llm, embeddings=real_embeddings, strictness=3
        )

        print("   Created both metrics for comparison")

        # Test cases for comparison
        test_cases = [
            {
                "user_input": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "description": "Simple factual answer",
            },
            {
                "user_input": "How does photosynthesis work?",
                "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
                "description": "Scientific explanation",
            },
            {
                "user_input": "What is machine learning?",
                "response": "I'm not entirely sure about the specific details of machine learning algorithms.",
                "description": "Noncommittal response",
            },
            {
                "user_input": "What is the weather like today?",
                "response": "I don't have access to real-time weather information.",
                "description": "Direct noncommittal answer",
            },
        ]

        differences = []

        for i, case in enumerate(test_cases):
            print(f"\n🧪 Testing case {i + 1}: {case['description']}")
            print(f"   Question: {case['user_input']}")
            print(f"   Response: {case['response'][:50]}...")

            # Test JSON prompt version (V2)
            json_result = await json_prompt_metric.ascore(
                user_input=case["user_input"], response=case["response"]
            )

            # Test string prompt version (Original)
            string_result = await string_prompt_metric.ascore(
                user_input=case["user_input"], response=case["response"]
            )

            json_score = json_result.value
            string_score = string_result.value
            diff = abs(json_score - string_score)

            differences.append(diff)

            print(f"   JSON Prompt (V2):  {json_score:.4f}")
            print(f"   String Prompt:     {string_score:.4f}")
            print(f"   Difference:        {diff:.4f}")

            # Both should be in valid range
            assert 0.0 <= json_score <= 1.0
            assert 0.0 <= string_score <= 1.0

            # Allow some tolerance for LLM randomness but scores should be reasonably close
            assert diff < 0.3, (
                f"Case {i + 1} ({case['description']}): Large difference: {json_score} vs {string_score}"
            )

            print("   ✅ Scores within tolerance!")

        # Overall statistics
        avg_diff = sum(differences) / len(differences)
        max_diff = max(differences)

        print("\n📊 Overall Results:")
        print(f"   Average difference: {avg_diff:.4f}")
        print(f"   Max difference:     {max_diff:.4f}")
        print("   All tests passed:   ✅")

        # Final assertions
        assert avg_diff < 0.2, f"Average difference too high: {avg_diff:.4f}"
        assert max_diff < 0.3, f"Maximum difference too high: {max_diff:.4f}"

        print("\n🎉 JSON Prompt vs String Prompt Comparison Complete!")
        print(
            "   • JSON prompt system (V2) produces similar results to string prompt system"
        )
        print(f"   • Average difference: {avg_diff:.4f} (acceptable)")
        print(f"   • Maximum difference: {max_diff:.4f} (within tolerance)")
        print(f"   • All {len(test_cases)} test cases passed ✅")
