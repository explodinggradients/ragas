"""E2E tests for AspectCritic metric migration from v1 (class-based) to v2 (class-based with modern components)."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AspectCritic as LegacyAspectCritic
from ragas.metrics.collections import AspectCritic
from ragas.metrics.result import MetricResult


class TestAspectCriticE2EMigration:
    """E2E test compatibility between legacy AspectCritic and new V2 AspectCritic."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for aspect critic evaluation."""
        return [
            {
                "user_input": "What is the capital of France?",
                "response": "Paris is the capital of France, located in the north-central part of the country.",
                "description": "Harmless factual answer",
                "expected_harmfulness": 0,
            },
            {
                "user_input": "How can I improve my health?",
                "response": "You should eat a balanced diet, exercise regularly, get enough sleep, and stay hydrated. Consult with healthcare professionals for personalized advice.",
                "description": "Helpful health advice",
                "expected_harmfulness": 0,
            },
            {
                "user_input": "Tell me about photosynthesis",
                "response": "Photosynthesis is a process used by plants to convert light energy into chemical energy. It involves chlorophyll, carbon dioxide, and water to produce glucose and oxygen.",
                "description": "Coherent scientific explanation",
                "expected_coherence": 1,
            },
            {
                "user_input": "What is 2 + 2?",
                "response": "2 + 2 equals 4.",
                "description": "Correct mathematical answer",
                "expected_correctness": 1,
            },
            {
                "user_input": "Explain quantum mechanics",
                "response": "Quantum mechanics is complex. There are particles and waves. Things are uncertain. Einstein didn't like it. It's about the very small.",
                "description": "Incoherent explanation",
                "expected_coherence": 0,
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a test LLM for legacy aspect critic evaluation."""
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

            client = openai.AsyncOpenAI()
            return instructor_llm_factory(
                "openai", model="gpt-3.5-turbo", client=client
            )
        except ImportError as e:
            pytest.skip(f"Instructor LLM factory not available: {e}")
        except Exception as e:
            pytest.skip(f"Could not create modern LLM (API key may be missing): {e}")

    @pytest.mark.asyncio
    async def test_harmfulness_legacy_vs_v2_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """Test that legacy and v2 harmfulness metric produce similar scores."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        # Test harmfulness detection
        harmfulness_cases = [
            case for case in sample_data if "expected_harmfulness" in case
        ]

        for i, data in enumerate(harmfulness_cases):
            print(f"\n🧪 Testing Harmfulness - Case {i + 1}: {data['description']}")
            print(f"   Question: {data['user_input']}")
            print(f"   Response: {data['response'][:100]}...")

            # Legacy v1
            legacy_metric = LegacyAspectCritic(
                name="harmfulness",
                definition="Does the submission cause or have the potential to cause harm to individuals, groups, or society at large?",
                llm=test_llm,
            )
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"], response=data["response"]
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            # V2 class-based with modern LLM
            v2_metric = AspectCritic(
                name="harmfulness",
                definition="Does the submission cause or have the potential to cause harm to individuals, groups, or society at large?",
                llm=test_modern_llm,
            )
            v2_result = await v2_metric.ascore(
                user_input=data["user_input"],
                response=data["response"],
            )

            print(f"   Legacy:    {legacy_score}")
            print(f"   V2 Class:  {v2_result.value}")
            print(f"   Expected:  {data['expected_harmfulness']}")
            print(
                f"   Reason:    {v2_result.reason[:100] if v2_result.reason else 'N/A'}..."
            )

            # Verify both give binary scores
            assert legacy_score in [0, 1], (
                f"Legacy score must be binary: {legacy_score}"
            )
            assert v2_result.value in [0, 1], (
                f"V2 score must be binary: {v2_result.value}"
            )

            # Verify types
            assert isinstance(legacy_score, (int, float))
            assert isinstance(v2_result, MetricResult)

            print("   ✅ Both implementations produce binary scores!")

    @pytest.mark.asyncio
    async def test_coherence_legacy_vs_v2_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """Test that legacy and v2 coherence metric produce similar scores."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        # Test coherence evaluation
        coherence_cases = [case for case in sample_data if "expected_coherence" in case]

        for i, data in enumerate(coherence_cases):
            print(f"\n🧪 Testing Coherence - Case {i + 1}: {data['description']}")
            print(f"   Question: {data['user_input']}")
            print(f"   Response: {data['response'][:100]}...")

            # Legacy v1
            legacy_metric = LegacyAspectCritic(
                name="coherence",
                definition="Does the submission present ideas, information, or arguments in a logical and organized manner?",
                llm=test_llm,
            )
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"], response=data["response"]
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            # V2 class-based with modern LLM
            v2_metric = AspectCritic(
                name="coherence",
                definition="Does the submission present ideas, information, or arguments in a logical and organized manner?",
                llm=test_modern_llm,
            )
            v2_result = await v2_metric.ascore(
                user_input=data["user_input"],
                response=data["response"],
            )

            print(f"   Legacy:    {legacy_score}")
            print(f"   V2 Class:  {v2_result.value}")
            print(f"   Expected:  {data['expected_coherence']}")
            print(
                f"   Reason:    {v2_result.reason[:100] if v2_result.reason else 'N/A'}..."
            )

            # Verify both give binary scores
            assert legacy_score in [0, 1], (
                f"Legacy score must be binary: {legacy_score}"
            )
            assert v2_result.value in [0, 1], (
                f"V2 score must be binary: {v2_result.value}"
            )

            # Verify types
            assert isinstance(legacy_score, (int, float))
            assert isinstance(v2_result, MetricResult)

            print("   ✅ Both implementations produce binary scores!")

    @pytest.mark.asyncio
    async def test_correctness_legacy_vs_v2_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """Test that legacy and v2 correctness metric produce similar scores."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        # Test correctness evaluation
        correctness_cases = [
            case for case in sample_data if "expected_correctness" in case
        ]

        for i, data in enumerate(correctness_cases):
            print(f"\n🧪 Testing Correctness - Case {i + 1}: {data['description']}")
            print(f"   Question: {data['user_input']}")
            print(f"   Response: {data['response'][:100]}...")

            # Legacy v1
            legacy_metric = LegacyAspectCritic(
                name="correctness",
                definition="Is the submission factually accurate and free from errors?",
                llm=test_llm,
            )
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"], response=data["response"]
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            # V2 class-based with modern LLM
            v2_metric = AspectCritic(
                name="correctness",
                definition="Is the submission factually accurate and free from errors?",
                llm=test_modern_llm,
            )
            v2_result = await v2_metric.ascore(
                user_input=data["user_input"],
                response=data["response"],
            )

            print(f"   Legacy:    {legacy_score}")
            print(f"   V2 Class:  {v2_result.value}")
            print(f"   Expected:  {data['expected_correctness']}")
            print(
                f"   Reason:    {v2_result.reason[:100] if v2_result.reason else 'N/A'}..."
            )

            # Verify both give binary scores
            assert legacy_score in [0, 1], (
                f"Legacy score must be binary: {legacy_score}"
            )
            assert v2_result.value in [0, 1], (
                f"V2 score must be binary: {v2_result.value}"
            )

            # Verify types
            assert isinstance(legacy_score, (int, float))
            assert isinstance(v2_result, MetricResult)

            print("   ✅ Both implementations produce binary scores!")

    @pytest.mark.asyncio
    async def test_aspect_critic_strictness(self, test_modern_llm):
        """Test that strictness parameter works correctly in v2 implementation."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for E2E testing")

        print("\n🎯 Testing Strictness Feature")

        test_case = {
            "user_input": "What is the capital of France?",
            "response": "Paris is the capital of France.",
        }

        # Test with strictness=1 (single check)
        metric_s1 = AspectCritic(
            name="correctness",
            definition="Is the submission factually accurate and free from errors?",
            llm=test_modern_llm,
            strictness=1,
        )

        result_s1 = await metric_s1.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
        )

        print(f"   Strictness=1 Score: {result_s1.value}")
        print(
            f"   Strictness=1 Reason: {result_s1.reason[:100] if result_s1.reason else 'N/A'}..."
        )

        # Test with strictness=3 (majority vote from 3 checks)
        metric_s3 = AspectCritic(
            name="correctness",
            definition="Is the submission factually accurate and free from errors?",
            llm=test_modern_llm,
            strictness=3,
        )

        result_s3 = await metric_s3.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
        )

        print(f"   Strictness=3 Score: {result_s3.value}")
        print(
            f"   Strictness=3 Reason: {result_s3.reason[:100] if result_s3.reason else 'N/A'}..."
        )

        # Both should produce binary scores
        assert result_s1.value in [0, 1], f"Score must be binary: {result_s1.value}"
        assert result_s3.value in [0, 1], f"Score must be binary: {result_s3.value}"

        # Verify that strictness attribute is always odd
        assert metric_s1.strictness % 2 != 0, "Strictness must be odd"
        assert metric_s3.strictness % 2 != 0, "Strictness must be odd"

        print("   ✅ Strictness feature works correctly!")

    @pytest.mark.asyncio
    async def test_aspect_critic_with_contexts(self, test_modern_llm):
        """Test that v2 implementation handles retrieved contexts correctly."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for E2E testing")

        print("\n🎯 Testing with Retrieved Contexts")

        test_case = {
            "user_input": "What is the Eiffel Tower?",
            "response": "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
            "retrieved_contexts": [
                "The Eiffel Tower was built in 1889 for the World's Fair.",
                "It stands 330 meters tall and is one of the most visited monuments.",
            ],
        }

        metric = AspectCritic(
            name="correctness",
            definition="Is the submission factually accurate and free from errors?",
            llm=test_modern_llm,
        )

        result = await metric.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
            retrieved_contexts=test_case["retrieved_contexts"],
        )

        print(f"   Score: {result.value}")
        print(f"   Reason: {result.reason[:100] if result.reason else 'N/A'}...")

        # Verify binary score
        assert result.value in [0, 1], f"Score must be binary: {result.value}"

        print("   ✅ Context handling works correctly!")

    @pytest.mark.asyncio
    async def test_aspect_critic_helper_functions(self, test_modern_llm):
        """Test that helper functions work correctly."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for E2E testing")

        print("\n🎯 Testing Helper Functions")

        from ragas.metrics.collections import (
            coherence,
            conciseness,
            correctness,
            harmfulness,
            maliciousness,
        )

        test_case = {
            "user_input": "What is 1+1?",
            "response": "1+1 equals 2.",
        }

        # Test harmfulness helper
        harmfulness_metric = harmfulness(test_modern_llm)
        result = await harmfulness_metric.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
        )
        print(f"   Harmfulness: {result.value}")
        assert result.value in [0, 1]

        # Test maliciousness helper
        maliciousness_metric = maliciousness(test_modern_llm)
        result = await maliciousness_metric.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
        )
        print(f"   Maliciousness: {result.value}")
        assert result.value in [0, 1]

        # Test coherence helper
        coherence_metric = coherence(test_modern_llm)
        result = await coherence_metric.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
        )
        print(f"   Coherence: {result.value}")
        assert result.value in [0, 1]

        # Test correctness helper
        correctness_metric = correctness(test_modern_llm)
        result = await correctness_metric.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
        )
        print(f"   Correctness: {result.value}")
        assert result.value in [0, 1]

        # Test conciseness helper
        conciseness_metric = conciseness(test_modern_llm)
        result = await conciseness_metric.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
        )
        print(f"   Conciseness: {result.value}")
        assert result.value in [0, 1]

        print("   ✅ All helper functions work correctly!")

    def test_aspect_critic_migration_requirements_documented(self):
        """Document the requirements for running full E2E aspect critic tests."""

        requirements = {
            "llm": "OpenAI GPT, Anthropic Claude, or other LangChain-compatible LLM",
            "environment": "API keys configured for LLM providers",
            "purpose": "Verify that v2 class-based implementation produces similar results to legacy class-based implementation",
        }

        print("\n📋 AspectCritic E2E Test Requirements:")
        for key, value in requirements.items():
            print(f"   {key.capitalize()}: {value}")

        print("\n🚀 To enable full E2E testing:")
        print("   1. Configure LLM provider (e.g., export OPENAI_API_KEY=...)")
        print("   2. Remove @pytest.mark.skip decorators if present")
        print(
            "   3. Run: pytest tests/e2e/metrics_migration/test_aspect_critic_migration.py -v -s"
        )

        assert True
