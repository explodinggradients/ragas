"""E2E tests for Answer Similarity metric migration from v1 to v2 (class-based)."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerSimilarity as LegacyAnswerSimilarity, MetricResult
from ragas.metrics.collections import AnswerSimilarity


class TestAnswerSimilarityE2EMigration:
    """E2E test compatibility between legacy AnswerSimilarity and new V2 AnswerSimilarity with automatic validation."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for answer similarity evaluation."""
        return [
            {
                "reference": "Paris is the capital of France.",
                "response": "The capital of France is Paris.",
                "description": "Semantically similar with word reordering",
            },
            {
                "reference": "Python is a high-level programming language known for its simplicity and readability.",
                "response": "Python is a programming language that emphasizes code readability.",
                "description": "Similar content with paraphrasing",
            },
            {
                "reference": "Machine learning is a subset of artificial intelligence.",
                "response": "Deep learning uses neural networks with multiple layers.",
                "description": "Related but different concepts",
            },
            {
                "reference": "The quick brown fox jumps over the lazy dog.",
                "response": "A slow red cat walks under the active mouse.",
                "description": "Different content with similar structure",
            },
            {
                "reference": "",
                "response": "Some response text",
                "description": "Empty reference",
            },
        ]

    @pytest.fixture
    def test_legacy_embeddings(self):
        """Create legacy embeddings for legacy implementation."""
        try:
            from ragas.embeddings.base import embedding_factory

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

            client = openai.AsyncOpenAI()

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
    async def test_legacy_answer_similarity_vs_v2_answer_similarity_e2e_compatibility(
        self,
        sample_data,
        test_legacy_embeddings,
        test_modern_embeddings,
    ):
        """E2E test that legacy and v2 implementations produce identical scores with real embeddings."""

        if test_legacy_embeddings is None or test_modern_embeddings is None:
            pytest.skip("Embeddings required for E2E testing")

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing Answer Similarity - Case {i + 1}: {data['description']}"
            )
            print(f"   Reference: {data['reference'][:50]}...")
            print(f"   Response:  {data['response'][:50]}...")

            legacy_answer_similarity = LegacyAnswerSimilarity(
                embeddings=test_legacy_embeddings
            )
            legacy_sample = SingleTurnSample(
                user_input="dummy",
                response=data["response"],
                reference=data["reference"],
            )
            legacy_score = await legacy_answer_similarity._single_turn_ascore(
                legacy_sample, None
            )

            v2_answer_similarity = AnswerSimilarity(embeddings=test_modern_embeddings)
            v2_answer_similarity_result = await v2_answer_similarity.ascore(
                reference=data["reference"],
                response=data["response"],
            )

            score_diff = abs(legacy_score - v2_answer_similarity_result.value)
            print(f"   Legacy:    {legacy_score:.6f}")
            print(f"   V2 Class:  {v2_answer_similarity_result.value:.6f}")
            print(f"   Diff:      {score_diff:.10f}")

            assert score_diff < 0.01, (
                f"Case {i + 1} ({data['description']}): Mismatch: {legacy_score} vs {v2_answer_similarity_result.value}"
            )

            assert isinstance(legacy_score, float)
            assert isinstance(v2_answer_similarity_result, MetricResult)
            assert 0.0 <= legacy_score <= 1.0
            assert 0.0 <= v2_answer_similarity_result.value <= 1.0

            print("   ✅ Scores match!")

    @pytest.mark.asyncio
    async def test_answer_similarity_with_threshold(
        self, test_legacy_embeddings, test_modern_embeddings
    ):
        """Test that both implementations correctly handle threshold parameter."""

        if test_legacy_embeddings is None or test_modern_embeddings is None:
            pytest.skip("Embeddings required for E2E testing")

        test_cases = [
            {
                "reference": "Paris is the capital of France.",
                "response": "The capital of France is Paris.",
                "threshold": 0.9,
                "description": "High similarity with high threshold",
            },
            {
                "reference": "Machine learning is a subset of artificial intelligence.",
                "response": "Deep learning uses neural networks.",
                "threshold": 0.5,
                "description": "Different content with medium threshold",
            },
        ]

        for case in test_cases:
            print(f"\n🎯 Testing threshold: {case['description']}")

            legacy_answer_similarity = LegacyAnswerSimilarity(
                embeddings=test_legacy_embeddings, threshold=case["threshold"]
            )
            legacy_sample = SingleTurnSample(
                user_input="dummy",
                response=case["response"],
                reference=case["reference"],
            )
            legacy_score = await legacy_answer_similarity._single_turn_ascore(
                legacy_sample, None
            )

            v2_answer_similarity = AnswerSimilarity(
                embeddings=test_modern_embeddings, threshold=case["threshold"]
            )
            v2_result = await v2_answer_similarity.ascore(
                reference=case["reference"],
                response=case["response"],
            )

            print(f"   Reference: {case['reference']}")
            print(f"   Response:  {case['response']}")
            print(f"   Threshold: {case['threshold']}")
            print(f"   Legacy:    {legacy_score:.6f}")
            print(f"   V2 Class:  {v2_result.value:.6f}")

            score_diff = abs(legacy_score - v2_result.value)
            assert score_diff < 1e-6, (
                f"Threshold test failed: {legacy_score} vs {v2_result.value}"
            )

            assert legacy_score in [0.0, 1.0]
            assert v2_result.value in [0.0, 1.0]

            print("   ✅ Threshold handling matches!")

    @pytest.mark.asyncio
    async def test_v2_class_batch_processing(self, sample_data, test_modern_embeddings):
        """Test V2 class-based AnswerSimilarity batch processing."""

        if test_modern_embeddings is None:
            pytest.skip("Modern embeddings required for V2 testing")

        metric = AnswerSimilarity(embeddings=test_modern_embeddings)

        batch_inputs = [
            {"reference": case["reference"], "response": case["response"]}
            for case in sample_data[:3]
        ]

        print(f"\n📦 Testing V2 class batch processing with {len(batch_inputs)} items:")

        results = await metric.abatch_score(batch_inputs)

        assert len(results) == len(batch_inputs)

        for i, (case, result) in enumerate(zip(sample_data[:3], results)):
            print(f"   Case {i + 1}: {result.value:.6f} - {case['description']}")
            assert isinstance(result.value, float)
            assert 0.0 <= result.value <= 1.0
            assert result.reason is None

        print("   ✅ V2 class batch processing works correctly!")

    def test_answer_similarity_migration_requirements_documented(self):
        """Document the requirements for running full E2E answer similarity tests."""

        requirements = {
            "embeddings": "OpenAI embeddings, HuggingFace embeddings, or similar",
            "environment": "API keys configured for embedding providers",
            "purpose": "Verify that v2 class-based implementation produces identical results to legacy implementation",
        }

        print("\n📋 Answer Similarity E2E Test Requirements:")
        for key, value in requirements.items():
            print(f"   {key.capitalize()}: {value}")

        print("\n🚀 To enable full E2E testing:")
        print("   1. Configure embedding provider (e.g., export OPENAI_API_KEY=...)")
        print("   2. Remove @pytest.mark.skip decorators")
        print(
            "   3. Run: pytest tests/e2e/metrics_migration/test_answer_similarity_migration.py -v -s"
        )

        assert True
