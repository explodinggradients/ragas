"""E2E tests for string metrics migration from v1 to v2."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import MetricResult
from ragas.metrics._string import (
    DistanceMeasure as LegacyDistanceMeasure,
    ExactMatch as LegacyExactMatch,
    NonLLMStringSimilarity as LegacyNonLLMStringSimilarity,
    StringPresence as LegacyStringPresence,
)
from ragas.metrics.collections import (
    DistanceMeasure,
    ExactMatch,
    NonLLMStringSimilarity,
    StringPresence,
)


class TestNonLLMStringSimilarityE2EMigration:
    """E2E test compatibility between legacy and new V2 implementations."""

    @pytest.fixture
    def sample_data(self):
        """Real-world sample reference and response texts for testing."""
        return [
            {
                "reference": "The cat sat on the mat",
                "response": "The cat sat on the mat",
                "description": "Exact match",
            },
            {
                "reference": "Hello World",
                "response": "Hallo World",
                "description": "Single character difference",
            },
            {
                "reference": "Python is a programming language",
                "response": "Python is a scripting language",
                "description": "Word substitution",
            },
            {
                "reference": "The capital of France is Paris",
                "response": "Paris is the capital of France",
                "description": "Word reordering",
            },
            {
                "reference": "Machine learning",
                "response": "Deep learning",
                "description": "Partial similarity",
            },
            {
                "reference": "test",
                "response": "test",
                "description": "Short exact match",
            },
            {
                "reference": "abc",
                "response": "xyz",
                "description": "Completely different",
            },
            {
                "reference": "",
                "response": "Some text",
                "description": "Empty reference",
            },
            {
                "reference": "Some text",
                "response": "",
                "description": "Empty response",
            },
        ]

    @pytest.mark.asyncio
    async def test_legacy_vs_v2_class_e2e_compatibility_levenshtein(self, sample_data):
        """E2E test that legacy and v2 class implementations produce identical scores (Levenshtein)."""

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing NonLLMStringSimilarity (Levenshtein) - Case {i + 1}: {data['description']}"
            )
            print(f"   Reference: '{data['reference']}'")
            print(f"   Response:  '{data['response']}'")

            legacy_metric = LegacyNonLLMStringSimilarity(
                distance_measure=LegacyDistanceMeasure.LEVENSHTEIN
            )
            legacy_sample = SingleTurnSample(
                user_input="dummy",
                response=data["response"],
                reference=data["reference"],
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            v2_class_metric = NonLLMStringSimilarity(
                distance_measure=DistanceMeasure.LEVENSHTEIN
            )
            v2_class_result = await v2_class_metric.ascore(
                reference=data["reference"],
                response=data["response"],
            )

            class_diff = abs(legacy_score - v2_class_result.value)

            print(f"   Legacy:      {legacy_score:.6f}")
            print(f"   V2 Class:    {v2_class_result.value:.6f}")
            print(f"   Diff:        {class_diff:.10f}")

            assert class_diff < 1e-10, (
                f"Case {i + 1} ({data['description']}): Levenshtein mismatch: "
                f"{legacy_score} != {v2_class_result.value}"
            )

            assert isinstance(legacy_score, float)
            assert isinstance(v2_class_result, MetricResult)

            print("   ✅ Legacy and V2 class produce identical scores!")

    @pytest.mark.asyncio
    async def test_legacy_vs_v2_class_e2e_compatibility_jaro_winkler(self, sample_data):
        """E2E test that legacy and v2 class implementations produce identical scores (Jaro-Winkler)."""

        for i, data in enumerate(sample_data[:5]):
            print(
                f"\n🧪 Testing NonLLMStringSimilarity (Jaro-Winkler) - Case {i + 1}: {data['description']}"
            )
            print(f"   Reference: '{data['reference']}'")
            print(f"   Response:  '{data['response']}'")

            legacy_metric = LegacyNonLLMStringSimilarity(
                distance_measure=LegacyDistanceMeasure.JARO_WINKLER
            )
            legacy_sample = SingleTurnSample(
                user_input="dummy",
                response=data["response"],
                reference=data["reference"],
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            v2_class_metric = NonLLMStringSimilarity(
                distance_measure=DistanceMeasure.JARO_WINKLER
            )
            v2_class_result = await v2_class_metric.ascore(
                reference=data["reference"],
                response=data["response"],
            )

            class_diff = abs(legacy_score - v2_class_result.value)

            print(f"   Legacy:      {legacy_score:.6f}")
            print(f"   V2 Class:    {v2_class_result.value:.6f}")
            print(f"   Diff:        {class_diff:.10f}")

            assert class_diff < 1e-10, (
                f"Case {i + 1} ({data['description']}): Jaro-Winkler mismatch: "
                f"{legacy_score} != {v2_class_result.value}"
            )

            assert isinstance(legacy_score, float)
            assert isinstance(v2_class_result, MetricResult)

            print("   ✅ Legacy and V2 class produce identical scores!")

    @pytest.mark.asyncio
    async def test_all_distance_measures(self):
        """Test that all distance measures work correctly in v2."""

        print("\n🔧 Testing all distance measures:")

        reference = "The quick brown fox"
        response = "The quick brown dog"

        for measure in DistanceMeasure:
            metric = NonLLMStringSimilarity(distance_measure=measure)
            result = await metric.ascore(reference=reference, response=response)

            print(f"   {measure.value:15s}: {result.value:.6f}")

            assert isinstance(result.value, float)
            assert 0.0 <= result.value <= 1.0

        print("   ✅ All distance measures work correctly!")

    @pytest.mark.asyncio
    async def test_performance_comparison(self, sample_data):
        """Compare performance characteristics between legacy and v2 class."""
        import time

        test_case = sample_data[3]

        print("\n⚡ Performance test: NonLLMStringSimilarity")

        legacy_metric = LegacyNonLLMStringSimilarity()
        legacy_sample = SingleTurnSample(
            user_input="dummy",
            response=test_case["response"],
            reference=test_case["reference"],
        )

        start_time = time.time()
        legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)
        legacy_time = time.time() - start_time

        v2_class_metric = NonLLMStringSimilarity()
        start_time = time.time()
        v2_class_result = await v2_class_metric.ascore(
            reference=test_case["reference"],
            response=test_case["response"],
        )
        v2_class_time = time.time() - start_time

        print(f"   Legacy:      {legacy_time:.4f}s → {legacy_score:.6f}")
        print(f"   V2 Class:    {v2_class_time:.4f}s → {v2_class_result.value:.6f}")

        assert abs(legacy_score - v2_class_result.value) < 1e-10
        assert isinstance(legacy_score, float)
        assert isinstance(v2_class_result, MetricResult)

    @pytest.mark.asyncio
    async def test_v2_class_no_components_needed(self):
        """Test that V2 class-based NonLLMStringSimilarity doesn't require LLM or embeddings."""

        print("\n🔧 Testing V2 NonLLMStringSimilarity component requirements:")

        metric = NonLLMStringSimilarity()

        print(f"   has llm attr: {hasattr(metric, 'llm')}")
        print(f"   has embeddings attr: {hasattr(metric, 'embeddings')}")

        result = await metric.ascore(
            reference="The capital of France is Paris.",
            response="Paris is the capital of France.",
        )

        print(f"   Score: {result.value:.6f}")

        assert not hasattr(metric, "llm") or metric.__dict__.get("llm") is None
        assert (
            not hasattr(metric, "embeddings")
            or metric.__dict__.get("embeddings") is None
        )
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0

        print("   ✅ V2 NonLLMStringSimilarity works without LLM/embeddings!")

    @pytest.mark.asyncio
    async def test_v2_class_batch_processing(self, sample_data):
        """Test V2 class-based NonLLMStringSimilarity batch processing."""

        metric = NonLLMStringSimilarity()

        batch_inputs = [
            {"reference": case["reference"], "response": case["response"]}
            for case in sample_data[:4]
        ]

        print(f"\n📦 Testing V2 class batch processing with {len(batch_inputs)} items:")

        results = await metric.abatch_score(batch_inputs)

        assert len(results) == len(batch_inputs)

        for i, (case, result) in enumerate(zip(sample_data[:4], results)):
            print(f"   Case {i + 1}: {result.value:.6f} - {case['description']}")
            assert isinstance(result.value, float)
            assert -1e-10 <= result.value <= 1.0 + 1e-10
            assert result.reason is None

        print("   ✅ V2 class batch processing works correctly!")

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases like empty strings."""

        print("\n🔍 Testing edge cases:")

        metric = NonLLMStringSimilarity()

        cases = [
            ("", "", "Both empty"),
            ("test", "", "Empty response"),
            ("", "test", "Empty reference"),
            ("a", "a", "Single character match"),
        ]

        for ref, resp, desc in cases:
            result = await metric.ascore(reference=ref, response=resp)
            print(f"   {desc:25s}: {result.value:.6f}")
            assert isinstance(result.value, float)
            assert 0.0 <= result.value <= 1.0

        print("   ✅ Edge cases handled correctly!")


class TestExactMatchE2EMigration:
    """E2E test compatibility between legacy ExactMatch and new V2 implementations."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for ExactMatch testing."""
        return [
            {
                "reference": "Hello World",
                "response": "Hello World",
                "description": "Exact match",
            },
            {
                "reference": "Hello World",
                "response": "hello world",
                "description": "Case mismatch",
            },
            {
                "reference": "Test",
                "response": "Test ",
                "description": "Trailing space",
            },
            {
                "reference": "",
                "response": "",
                "description": "Both empty",
            },
            {
                "reference": "abc",
                "response": "xyz",
                "description": "Completely different",
            },
        ]

    @pytest.mark.asyncio
    async def test_legacy_vs_v2_class_e2e_compatibility(self, sample_data):
        """E2E test that legacy and v2 class implementations produce identical scores."""

        for i, data in enumerate(sample_data):
            print(f"\n🧪 Testing ExactMatch - Case {i + 1}: {data['description']}")
            print(f"   Reference: '{data['reference']}'")
            print(f"   Response:  '{data['response']}'")

            legacy_metric = LegacyExactMatch()
            legacy_sample = SingleTurnSample(
                user_input="dummy",
                response=data["response"],
                reference=data["reference"],
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            v2_class_metric = ExactMatch()
            v2_class_result = await v2_class_metric.ascore(
                reference=data["reference"],
                response=data["response"],
            )

            class_diff = abs(legacy_score - v2_class_result.value)

            print(f"   Legacy:      {legacy_score:.6f}")
            print(f"   V2 Class:    {v2_class_result.value:.6f}")
            print(f"   Diff:        {class_diff:.10f}")

            assert class_diff < 1e-10, (
                f"Case {i + 1} ({data['description']}): ExactMatch mismatch: "
                f"{legacy_score} != {v2_class_result.value}"
            )

            assert isinstance(legacy_score, float)
            assert isinstance(v2_class_result, MetricResult)

            print("   ✅ Legacy and V2 class produce identical scores!")


class TestStringPresenceE2EMigration:
    """E2E test compatibility between legacy StringPresence and new V2 implementations."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for StringPresence testing."""
        return [
            {
                "reference": "Paris",
                "response": "The capital of France is Paris.",
                "description": "String present",
            },
            {
                "reference": "cat",
                "response": "The cat sat on the mat",
                "description": "String present in middle",
            },
            {
                "reference": "dog",
                "response": "The cat sat on the mat",
                "description": "String not present",
            },
            {
                "reference": "Hello",
                "response": "Hello World",
                "description": "String at start",
            },
            {
                "reference": "World",
                "response": "Hello World",
                "description": "String at end",
            },
            {
                "reference": "",
                "response": "Some text",
                "description": "Empty reference",
            },
            {
                "reference": "test",
                "response": "",
                "description": "Empty response",
            },
        ]

    @pytest.mark.asyncio
    async def test_legacy_vs_v2_class_e2e_compatibility(self, sample_data):
        """E2E test that legacy and v2 class implementations produce identical scores."""

        for i, data in enumerate(sample_data):
            print(f"\n🧪 Testing StringPresence - Case {i + 1}: {data['description']}")
            print(f"   Reference: '{data['reference']}'")
            print(f"   Response:  '{data['response']}'")

            legacy_metric = LegacyStringPresence()
            legacy_sample = SingleTurnSample(
                user_input="dummy",
                response=data["response"],
                reference=data["reference"],
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            v2_class_metric = StringPresence()
            v2_class_result = await v2_class_metric.ascore(
                reference=data["reference"],
                response=data["response"],
            )

            class_diff = abs(legacy_score - v2_class_result.value)

            print(f"   Legacy:      {legacy_score:.6f}")
            print(f"   V2 Class:    {v2_class_result.value:.6f}")
            print(f"   Diff:        {class_diff:.10f}")

            assert class_diff < 1e-10, (
                f"Case {i + 1} ({data['description']}): StringPresence mismatch: "
                f"{legacy_score} != {v2_class_result.value}"
            )

            assert isinstance(legacy_score, float)
            assert isinstance(v2_class_result, MetricResult)

            print("   ✅ Legacy and V2 class produce identical scores!")
