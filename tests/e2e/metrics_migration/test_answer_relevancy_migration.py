"""E2E tests for Answer Relevancy metric migration from v1 (class-based) to v2 (decorator-based)."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerRelevancy, MetricResult
from ragas.metrics.v2 import answer_relevancy


class TestAnswerRelevancyE2EMigration:
    """E2E test compatibility between legacy AnswerRelevancy class and new answer_relevancy decorator."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for answer relevancy evaluation."""
        return [
            {
                "user_input": "What is the capital of France?",
                "response": "The capital of France is Paris, which is located in the north-central part of the country and serves as the political, economic, and cultural center.",
                "description": "Direct answer with extra context",
            },
            {
                "user_input": "How does photosynthesis work?",
                "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
                "description": "Scientific explanation",
            },
            {
                "user_input": "What is the weather like today?",
                "response": "I don't have access to real-time weather data, so I cannot tell you what the weather is like today.",
                "description": "Noncommittal response - should get low score",
            },
            {
                "user_input": "Explain quantum computing",
                "response": "Classical computers use bits, but quantum computers are different. There are many complex theories involved.",
                "description": "Vague/incomplete answer",
            },
            {
                "user_input": "What is 2 + 2?",
                "response": "2 + 2 equals 4.",
                "description": "Simple direct answer",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a test LLM for legacy answer relevancy evaluation."""
        # Use legacy llm_factory for legacy implementation
        try:
            from ragas.llms.base import llm_factory

            return llm_factory("gpt-3.5-turbo")
        except ImportError as e:
            pytest.skip(f"LLM factory not available: {e}")
        except Exception as e:
            pytest.skip(f"Could not create LLM (API key may be missing): {e}")

    @pytest.fixture
    def test_modern_llm(self):
        """Create a modern instructor LLM for v2 implementation."""
        try:
            import openai

            from ragas.llms.base import instructor_llm_factory

            client = openai.OpenAI()
            return instructor_llm_factory(
                "openai", model="gpt-3.5-turbo", client=client
            )
        except ImportError as e:
            pytest.skip(f"Instructor LLM factory not available: {e}")
        except Exception as e:
            pytest.skip(f"Could not create modern LLM (API key may be missing): {e}")

    @pytest.fixture
    def test_legacy_embeddings(self):
        """Create legacy embeddings for legacy implementation."""
        try:
            from ragas.embeddings.base import embedding_factory

            # Use legacy interface for legacy implementation
            return embedding_factory("text-embedding-ada-002")
        except ImportError as e:
            pytest.skip(f"Embedding factory not available: {e}")
        except Exception as e:
            pytest.skip(
                f"Could not create legacy embeddings (API key may be missing): {e}"
            )

    @pytest.fixture
    def test_modern_embeddings(self):
        """Create modern embeddings for v2 implementation."""
        try:
            import openai

            from ragas.embeddings.base import embedding_factory

            # Create OpenAI client
            client = openai.OpenAI()

            # Use modern interface with explicit provider and client
            return embedding_factory(
                provider="openai",
                model="text-embedding-ada-002",
                client=client,
                interface="modern",
            )
        except ImportError as e:
            pytest.skip(f"OpenAI or embedding factory not available: {e}")
        except Exception as e:
            pytest.skip(
                f"Could not create modern embeddings (API key may be missing): {e}"
            )

    @pytest.mark.asyncio
    async def test_legacy_answer_relevancy_vs_v2_answer_relevancy_e2e_compatibility(
        self,
        sample_data,
        test_llm,
        test_modern_llm,
        test_legacy_embeddings,
        test_modern_embeddings,
    ):
        """E2E test that legacy and v2 implementations produce similar scores with real LLM."""

        if (
            test_llm is None
            or test_modern_llm is None
            or test_legacy_embeddings is None
            or test_modern_embeddings is None
        ):
            pytest.skip("LLM and embeddings required for E2E testing")

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing Answer Relevancy - Case {i + 1}: {data['description']}"
            )
            print(f"   Question: {data['user_input']}")
            print(f"   Response: {data['response'][:100]}...")

            # Legacy v1 with legacy embeddings
            legacy_answer_relevancy = AnswerRelevancy(
                llm=test_llm, embeddings=test_legacy_embeddings
            )
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"], response=data["response"]
            )
            legacy_score = await legacy_answer_relevancy._single_turn_ascore(
                legacy_sample, None
            )

            # V2 with modern embeddings and modern LLM - call the function directly
            v2_answer_relevancy_result = await answer_relevancy(
                user_input=data["user_input"],
                response=data["response"],
                llm=test_modern_llm,
                embeddings=test_modern_embeddings,
            )

            # Results might not be exactly identical due to LLM randomness, but should be close
            score_diff = abs(legacy_score - v2_answer_relevancy_result.value)
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_answer_relevancy_result.value:.6f}")
            print(f"   Diff:   {score_diff:.6f}")

            # Allow some tolerance for LLM randomness but scores should be reasonably close
            assert score_diff < 0.2, (
                f"Case {i + 1} ({data['description']}): Large difference: {legacy_score} vs {v2_answer_relevancy_result.value}"
            )

            # Verify types
            assert isinstance(legacy_score, float)
            assert isinstance(v2_answer_relevancy_result, MetricResult)
            assert 0.0 <= legacy_score <= 1.0
            assert 0.0 <= v2_answer_relevancy_result.value <= 1.0

            print("   ✅ Scores within tolerance!")

    @pytest.mark.asyncio
    async def test_answer_relevancy_noncommittal_detection(
        self, test_llm, test_modern_llm, test_legacy_embeddings, test_modern_embeddings
    ):
        """Test that both implementations correctly detect noncommittal answers."""

        if (
            test_llm is None
            or test_modern_llm is None
            or test_legacy_embeddings is None
            or test_modern_embeddings is None
        ):
            pytest.skip("LLM and embeddings required for E2E testing")

        # Test cases specifically for noncommittal detection
        test_cases = [
            {
                "user_input": "What is the population of Tokyo?",
                "response": "I don't know the exact population of Tokyo.",
                "expected_low": True,
                "description": "Clear noncommittal",
            },
            {
                "user_input": "What is the population of Tokyo?",
                "response": "Tokyo has a population of approximately 14 million people in the metropolitan area.",
                "expected_low": False,
                "description": "Committal answer",
            },
        ]

        for case in test_cases:
            print(f"\n🎯 Testing noncommittal detection: {case['description']}")

            # Legacy with legacy embeddings
            legacy_answer_relevancy = AnswerRelevancy(
                llm=test_llm, embeddings=test_legacy_embeddings
            )
            legacy_sample = SingleTurnSample(
                user_input=case["user_input"], response=case["response"]
            )
            legacy_score = await legacy_answer_relevancy._single_turn_ascore(
                legacy_sample, None
            )

            # V2 with modern embeddings and modern LLM - call the function directly
            v2_result = await answer_relevancy(
                user_input=case["user_input"],
                response=case["response"],
                llm=test_modern_llm,
                embeddings=test_modern_embeddings,
            )

            print(f"   Response: {case['response']}")
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")

            if case["expected_low"]:
                # Noncommittal answers should get low scores (close to 0)
                assert legacy_score < 0.1, (
                    f"Legacy should detect noncommittal: {legacy_score}"
                )
                assert v2_result.value < 0.1, (
                    f"V2 should detect noncommittal: {v2_result.value}"
                )
                print("   ✅ Both detected noncommittal (low scores)")
            else:
                # Committal answers should get reasonable scores
                assert legacy_score > 0.3, (
                    f"Legacy should score committal higher: {legacy_score}"
                )
                assert v2_result.value > 0.3, (
                    f"V2 should score committal higher: {v2_result.value}"
                )
                print("   ✅ Both scored committal answer reasonably")

    def test_answer_relevancy_migration_requirements_documented(self):
        """Document the requirements for running full E2E answer relevancy tests."""

        requirements = {
            "llm": "OpenAI GPT, Anthropic Claude, or other LangChain-compatible LLM",
            "embeddings": "OpenAI embeddings, HuggingFace embeddings, or similar",
            "environment": "API keys configured for LLM and embedding providers",
            "purpose": "Verify that v2 decorator implementation produces similar results to legacy class-based implementation",
        }

        # To run full E2E tests, users would need to:
        # 1. Configure LLM (e.g., export OPENAI_API_KEY=...)
        # 2. Configure embeddings
        # 3. Remove @pytest.mark.skip decorators
        # 4. Run: pytest tests/e2e/metrics_migration/test_answer_relevancy_migration.py -v -s

        print("\n📋 Answer Relevancy E2E Test Requirements:")
        for key, value in requirements.items():
            print(f"   {key.capitalize()}: {value}")

        print("\n🚀 To enable full E2E testing:")
        print("   1. Configure LLM provider (e.g., export OPENAI_API_KEY=...)")
        print("   2. Configure embeddings provider")
        print("   3. Remove @pytest.mark.skip decorators")
        print(
            "   4. Run: pytest tests/e2e/metrics_migration/test_answer_relevancy_migration.py -v -s"
        )

        assert True
