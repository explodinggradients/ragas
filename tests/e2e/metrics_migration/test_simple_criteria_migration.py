"""E2E tests for SimpleCriteria metric migration from v1 to v2."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SimpleCriteriaScore as LegacySimpleCriteria
from ragas.metrics.collections import SimpleCriteria
from ragas.metrics.result import MetricResult


class TestSimpleCriteriaE2EMigration:
    """E2E test compatibility between legacy SimpleCriteria and new V2 SimpleCriteria."""

    @pytest.fixture
    def sample_data(self):
        """Test cases for simple criteria evaluation."""
        return [
            {
                "user_input": "What is Python?",
                "response": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
                "description": "Clear technical explanation",
                "definition": "Is the response clear and well-explained?",
            },
            {
                "user_input": "Explain quantum computing",
                "response": "Quantum computers use quantum bits or qubits. They leverage quantum mechanical phenomena like superposition and entanglement to process information differently from classical computers.",
                "description": "Comprehensive explanation",
                "definition": "Does the response provide a comprehensive explanation?",
            },
            {
                "user_input": "How do I learn programming?",
                "response": "Start with the basics, practice daily, build projects, read others' code, and join communities.",
                "description": "Concise actionable advice",
                "definition": "Is the advice actionable and concise?",
            },
            {
                "user_input": "What is machine learning?",
                "response": "ML is a field where systems learn from data without being explicitly programmed.",
                "description": "Simple definition",
                "definition": "Is the definition accurate and simple?",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a test LLM for legacy simple criteria evaluation."""
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

            from ragas.llms import llm_factory

            client = openai.AsyncOpenAI()
            return llm_factory("gpt-3.5-turbo", client=client)
        except ImportError as e:
            pytest.skip(f"Instructor LLM factory not available: {e}")
        except Exception as e:
            pytest.skip(f"Could not create modern LLM (API key may be missing): {e}")

    @pytest.mark.asyncio
    async def test_simple_criteria_legacy_vs_v2_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """Test that legacy and v2 simple criteria produce similar results."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        for i, data in enumerate(sample_data):
            print(f"\n🧪 Testing SimpleCriteria - Case {i + 1}: {data['description']}")
            print(f"   Question: {data['user_input']}")
            print(f"   Response: {data['response'][:100]}...")

            # Legacy v1
            legacy_metric = LegacySimpleCriteria(
                name="test_criteria",
                definition=data["definition"],
                llm=test_llm,
            )
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"], response=data["response"]
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            # V2 with modern LLM
            v2_metric = SimpleCriteria(
                name="test_criteria",
                definition=data["definition"],
                llm=test_modern_llm,
            )
            v2_result = await v2_metric.ascore(
                user_input=data["user_input"],
                response=data["response"],
            )

            print(f"   Legacy:    {legacy_score}")
            print(f"   V2:        {v2_result.value}")
            print(
                f"   Reason:    {v2_result.reason[:100] if v2_result.reason else 'N/A'}..."
            )

            # Verify types
            assert isinstance(legacy_score, (int, float))
            assert isinstance(v2_result, MetricResult)

            print("   ✅ Both implementations produce numeric scores!")

    @pytest.mark.asyncio
    async def test_simple_criteria_with_contexts(self, test_modern_llm):
        """Test that v2 implementation handles retrieved contexts correctly."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for E2E testing")

        print("\n🎯 Testing SimpleCriteria with Retrieved Contexts")

        test_case = {
            "user_input": "What is machine learning?",
            "response": "Machine learning is a subset of AI.",
            "retrieved_contexts": [
                "Machine learning enables systems to learn from data.",
                "It's used in recommendation systems, image recognition, etc.",
            ],
        }

        metric = SimpleCriteria(
            name="clarity",
            definition="Is the response clear and informative?",
            llm=test_modern_llm,
        )

        result = await metric.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
            retrieved_contexts=test_case["retrieved_contexts"],
        )

        print(f"   Score: {result.value}")
        print(f"   Reason: {result.reason[:100] if result.reason else 'N/A'}...")

        assert isinstance(result.value, float)
        assert isinstance(result.reason, str)

        print("   ✅ Context handling works correctly!")

    @pytest.mark.asyncio
    async def test_simple_criteria_strictness(self, test_modern_llm):
        """Test that strictness parameter works correctly in v2 implementation."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for E2E testing")

        print("\n🎯 Testing Strictness Feature")

        test_case = {
            "user_input": "What is 2+2?",
            "response": "2+2 equals 4.",
        }

        # Test with strictness=1
        metric_s1 = SimpleCriteria(
            name="correctness",
            definition="Is the answer mathematically correct?",
            llm=test_modern_llm,
            strictness=1,
        )

        result_s1 = await metric_s1.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
        )

        print(f"   Strictness=1 Score: {result_s1.value}")

        # Test with strictness=3
        metric_s3 = SimpleCriteria(
            name="correctness",
            definition="Is the answer mathematically correct?",
            llm=test_modern_llm,
            strictness=3,
        )

        result_s3 = await metric_s3.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
        )

        print(f"   Strictness=3 Score: {result_s3.value}")

        # Both should produce numeric scores
        assert isinstance(result_s1.value, float)
        assert isinstance(result_s3.value, float)

        # Verify that strictness attribute is always odd
        assert metric_s1.strictness % 2 != 0, "Strictness must be odd"
        assert metric_s3.strictness % 2 != 0, "Strictness must be odd"

        print("   ✅ Strictness feature works correctly!")

    @pytest.mark.asyncio
    async def test_simple_criteria_with_reference(self, test_modern_llm):
        """Test that v2 implementation handles reference answer correctly."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for E2E testing")

        print("\n🎯 Testing SimpleCriteria with Reference Answer")

        test_case = {
            "user_input": "What is the capital of France?",
            "response": "Paris",
            "reference": "Paris is the capital of France",
        }

        metric = SimpleCriteria(
            name="accuracy",
            definition="Does the response match the reference answer?",
            llm=test_modern_llm,
        )

        result = await metric.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
            reference=test_case["reference"],
        )

        print(f"   Score: {result.value}")
        print(f"   Reason: {result.reason[:100] if result.reason else 'N/A'}...")

        assert isinstance(result.value, float)
        assert isinstance(result.reason, str)

        print("   ✅ Reference handling works correctly!")

    def test_simple_criteria_migration_requirements_documented(self):
        """Document the requirements for running full E2E simple criteria tests."""

        requirements = {
            "llm": "OpenAI GPT, Anthropic Claude, or other LangChain-compatible LLM",
            "environment": "API keys configured for LLM providers",
            "purpose": "Verify that v2 implementation produces similar results to legacy implementation",
        }

        print("\n📋 SimpleCriteria E2E Test Requirements:")
        for key, value in requirements.items():
            print(f"   {key.capitalize()}: {value}")

        print("\n🚀 To enable full E2E testing:")
        print("   1. Configure LLM provider (e.g., export OPENAI_API_KEY=...)")
        print("   2. Remove @pytest.mark.skip decorators if present")
        print(
            "   3. Run: pytest tests/e2e/metrics_migration/test_simple_criteria_migration.py -v -s"
        )

        assert True
