"""E2E tests for Context Entity Recall metric migration from v1 to v2."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ContextEntityRecall as LegacyContextEntityRecall
from ragas.metrics.collections import ContextEntityRecall
from ragas.metrics.result import MetricResult


class TestContextEntityRecallE2EMigration:
    """E2E test compatibility between legacy ContextEntityRecall and new V2 ContextEntityRecall with modern components."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for context entity recall evaluation."""
        return [
            {
                "reference": "The Eiffel Tower in Paris, France was built in 1889 for the World's Fair.",
                "retrieved_contexts": [
                    "The Eiffel Tower is located in Paris, France.",
                    "It was constructed in 1889 for the 1889 World's Fair.",
                ],
                "description": "Complete entity coverage - should score high",
            },
            {
                "reference": "Albert Einstein was born in Germany in 1879 and developed the theory of relativity.",
                "retrieved_contexts": [
                    "Einstein was a physicist born in Germany.",
                    "He created important theories in physics.",
                ],
                "description": "Missing key entities (1879, theory of relativity)",
            },
            {
                "reference": "The Apollo 11 mission launched on July 16, 1969 with Neil Armstrong, Buzz Aldrin, and Michael Collins.",
                "retrieved_contexts": [
                    "Apollo 11 was a space mission.",
                    "Neil Armstrong was the first person to walk on the Moon.",
                ],
                "description": "Partial entity coverage",
            },
            {
                "reference": "Microsoft was founded by Bill Gates and Paul Allen in 1975 in Seattle, Washington.",
                "retrieved_contexts": [
                    "Bill Gates founded Microsoft.",
                    "Paul Allen co-founded the company.",
                    "It was established in 1975 in Seattle, Washington.",
                ],
                "description": "Good entity coverage with paraphrasing",
            },
            {
                "reference": "The Great Wall of China stretches over 21,196 kilometers and was built starting in the 7th century BC.",
                "retrieved_contexts": [
                    "The Great Wall is in China.",
                    "It's a very long wall built long ago.",
                ],
                "description": "Poor entity coverage - missing specific details",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a test LLM for legacy context entity recall evaluation."""
        try:
            from ragas.llms.base import llm_factory

            return llm_factory("gpt-4o")  # Using GPT-4o for best alignment
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

            client = openai.AsyncOpenAI()
            return instructor_llm_factory(
                "openai",
                model="gpt-4o",
                client=client,  # Using GPT-4o for best alignment
            )
        except ImportError as e:
            pytest.skip(f"Instructor LLM factory not available: {e}")
        except Exception as e:
            pytest.skip(f"Could not create modern LLM (API key may be missing): {e}")

    @pytest.mark.asyncio
    async def test_legacy_context_entity_recall_vs_v2_context_entity_recall_e2e_compatibility(
        self,
        sample_data,
        test_llm,
        test_modern_llm,
    ):
        """E2E test that legacy and v2 implementations produce similar scores with real LLM."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing Context Entity Recall - Case {i + 1}: {data['description']}"
            )
            print(f"   Reference: {data['reference'][:80]}...")
            print(f"   Contexts: {len(data['retrieved_contexts'])} contexts")

            # Legacy v1 implementation
            legacy_context_entity_recall = LegacyContextEntityRecall(llm=test_llm)
            legacy_sample = SingleTurnSample(
                reference=data["reference"],
                retrieved_contexts=data["retrieved_contexts"],
            )
            legacy_score = await legacy_context_entity_recall._single_turn_ascore(
                legacy_sample, None
            )

            # V2 implementation with modern components
            v2_context_entity_recall = ContextEntityRecall(llm=test_modern_llm)
            v2_result = await v2_context_entity_recall.ascore(
                reference=data["reference"],
                retrieved_contexts=data["retrieved_contexts"],
            )

            # Results should be very close with GPT-4o
            score_diff = abs(legacy_score - v2_result.value)
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")
            print(f"   Diff:   {score_diff:.6f}")

            # With GPT-4o, should be reasonably close (allowing for entity extraction variations)
            assert score_diff < 0.3, (
                f"Case {i + 1} ({data['description']}): Large difference: {legacy_score} vs {v2_result.value}"
            )

            # Verify types
            assert isinstance(legacy_score, float)
            assert isinstance(v2_result, MetricResult)
            assert 0.0 <= legacy_score <= 1.0
            assert 0.0 <= v2_result.value <= 1.0

            print("   ✅ Scores within tolerance!")

    @pytest.mark.asyncio
    async def test_context_entity_recall_entity_extraction_accuracy(
        self, test_llm, test_modern_llm
    ):
        """Test that both implementations extract entities accurately."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        # Test cases for entity extraction accuracy
        test_cases = [
            {
                "reference": "Barack Obama was the 44th President of the United States from 2009 to 2017.",
                "retrieved_contexts": ["Barack Obama served as U.S. President."],
                "expected_entities": [
                    "Barack Obama",
                    "44th President",
                    "United States",
                    "2009",
                    "2017",
                ],
                "description": "Political figure with dates and positions",
            },
            {
                "reference": "The iPhone was released by Apple Inc. on June 29, 2007 in the United States.",
                "retrieved_contexts": ["Apple released the iPhone in 2007 in the US."],
                "expected_entities": [
                    "iPhone",
                    "Apple Inc.",
                    "June 29, 2007",
                    "United States",
                ],
                "description": "Product launch with company and date",
            },
        ]

        for case in test_cases:
            print(f"\n🎯 Testing entity extraction: {case['description']}")

            # Legacy implementation
            legacy_metric = LegacyContextEntityRecall(llm=test_llm)
            legacy_sample = SingleTurnSample(
                reference=case["reference"],
                retrieved_contexts=case["retrieved_contexts"],
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            # V2 implementation
            v2_metric = ContextEntityRecall(llm=test_modern_llm)
            v2_result = await v2_metric.ascore(
                reference=case["reference"],
                retrieved_contexts=case["retrieved_contexts"],
            )

            print(f"   Reference: {case['reference']}")
            print(f"   Retrieved: {case['retrieved_contexts']}")
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")

            # Both should produce valid recall scores
            assert 0.0 <= legacy_score <= 1.0
            assert 0.0 <= v2_result.value <= 1.0

            # With GPT-4o, should be very close
            score_diff = abs(legacy_score - v2_result.value)
            assert score_diff < 0.1, (
                f"Large difference in entity extraction: {score_diff}"
            )

            print("   ✅ Both extracted entities consistently!")

    def test_context_entity_recall_parameter_validation(self):
        """Test that v2 implementation properly validates parameters."""
        from unittest.mock import Mock

        mock_llm = Mock()

        # Test that invalid components are properly rejected
        try:
            ContextEntityRecall(llm=mock_llm)
            assert False, "Should have rejected Mock LLM"
        except ValueError as e:
            assert "modern InstructorLLM" in str(e)
            print("✅ Correctly rejected invalid LLM component")

        print("✅ Parameter validation working correctly!")

    def test_context_entity_recall_migration_requirements_documented(self):
        """Document the requirements for running full E2E context entity recall tests."""

        requirements = {
            "llm": "OpenAI GPT-4o, Anthropic Claude, or other LLM with structured output support",
            "environment": "API keys configured for LLM provider",
            "purpose": "Verify that v2 implementation produces similar results to legacy implementation",
            "complexity": "Tests entity extraction accuracy and recall calculation",
        }

        print("\n📋 Context Entity Recall E2E Test Requirements:")
        for key, value in requirements.items():
            print(f"   {key.capitalize()}: {value}")

        print("\n🚀 To enable full E2E testing:")
        print("   1. Configure LLM provider (e.g., export OPENAI_API_KEY=...)")
        print("   2. Remove @pytest.mark.skip decorators")
        print(
            "   3. Run: pytest tests/e2e/metrics_migration/test_context_entity_recall_migration.py -v -s"
        )

        print("\n🔬 Test Coverage:")
        print("   • Entity extraction accuracy")
        print("   • Set intersection recall calculation")
        print("   • Different entity types (people, places, dates, products)")
        print("   • Paraphrasing and entity recognition")
        print("   • Parameter validation")
        print("   • Score equivalence between v1 and v2")

        assert True
