"""E2E tests for ROUGE score metric migration from v1 to v2 (function and class-based)."""

import typing as t

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import MetricResult, RougeScore as LegacyRougeScore
from ragas.metrics.v2 import RougeScore

# Type aliases for better type checking
RougeType = t.Literal["rouge1", "rougeL"]
RougeMode = t.Literal["fmeasure", "precision", "recall"]


class TestRougeE2EMigration:
    """E2E test compatibility between legacy RougeScore and new V2 implementations (function and class-based)."""

    @pytest.fixture
    def sample_data(self):
        """Real-world sample reference and response texts for testing."""
        return [
            {
                "reference": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
                "response": "Python is a programming language that emphasizes code readability and was developed by Guido van Rossum in 1991.",
                "description": "Similar content with paraphrasing",
            },
            {
                "reference": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "response": "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
                "description": "Related but different content",
            },
            {
                "reference": "The capital of France is Paris, which is also the most populous city in the country.",
                "response": "Paris is the capital and largest city of France.",
                "description": "Concise vs detailed",
            },
            {
                "reference": "",
                "response": "Some response text",
                "description": "Empty reference",
            },
            {
                "reference": "Some reference text",
                "response": "",
                "description": "Empty response",
            },
        ]

    @pytest.mark.parametrize(
        "rouge_type,mode",
        [
            ("rouge1", "fmeasure"),
            ("rouge1", "precision"),
            ("rouge1", "recall"),
            ("rougeL", "fmeasure"),
            ("rougeL", "precision"),
            ("rougeL", "recall"),
        ],
    )
    @pytest.mark.asyncio
    async def test_legacy_vs_v2_class_e2e_compatibility(
        self, sample_data, rouge_type: RougeType, mode: RougeMode
    ):
        """E2E test that legacy and v2 class implementations produce identical scores."""

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing {rouge_type} {mode} - Case {i + 1}: {data['description']}"
            )
            print(f"   Reference: {data['reference'][:50]}...")
            print(f"   Response:  {data['response'][:50]}...")

            # Legacy v1
            legacy_rouge_score = LegacyRougeScore(rouge_type=rouge_type, mode=mode)
            legacy_sample = SingleTurnSample(
                user_input="dummy",
                response=data["response"],
                reference=data["reference"],
            )
            legacy_score = await legacy_rouge_score._single_turn_ascore(
                legacy_sample, None
            )

            # V2 class-based
            v2_class_metric = RougeScore(rouge_type=rouge_type, mode=mode)
            v2_class_result = await v2_class_metric.ascore(
                reference=data["reference"],
                response=data["response"],
            )

            # Verify exact matches
            class_diff = abs(legacy_score - v2_class_result.value)

            print(f"   Legacy:      {legacy_score:.6f}")
            print(f"   V2 Class:    {v2_class_result.value:.6f}")
            print(f"   Diff:        {class_diff:.10f}")

            assert class_diff < 1e-10, (
                f"Case {i + 1} ({data['description']}): {rouge_type} {mode} class mismatch: {legacy_score} != {v2_class_result.value}"
            )

            # Verify types (legacy can return int 0 or float)
            assert isinstance(legacy_score, (int, float))
            assert isinstance(v2_class_result, MetricResult)

            print("   ✅ Legacy and V2 class produce identical scores!")

    @pytest.mark.asyncio
    async def test_rouge_score_performance_comparison(self, sample_data):
        """Compare performance characteristics between legacy and v2 class."""
        import time

        # Test with multiple configurations
        configs: t.List[t.Tuple[RougeType, RougeMode]] = [
            ("rouge1", "fmeasure"),
            ("rougeL", "fmeasure"),
        ]
        test_case = sample_data[0]  # Use first realistic test case

        for rouge_type, mode in configs:
            print(f"\n⚡ Performance test: {rouge_type} {mode}")

            # Legacy timing
            legacy_rouge_score = LegacyRougeScore(rouge_type=rouge_type, mode=mode)
            legacy_sample = SingleTurnSample(
                user_input="dummy",
                response=test_case["response"],
                reference=test_case["reference"],
            )

            start_time = time.time()
            legacy_score = await legacy_rouge_score._single_turn_ascore(
                legacy_sample, None
            )
            legacy_time = time.time() - start_time

            # V2 class timing
            v2_class_metric = RougeScore(rouge_type=rouge_type, mode=mode)
            start_time = time.time()
            v2_class_result = await v2_class_metric.ascore(
                reference=test_case["reference"],
                response=test_case["response"],
            )
            v2_class_time = time.time() - start_time

            print(f"   Legacy:      {legacy_time:.4f}s → {legacy_score:.6f}")
            print(f"   V2 Class:    {v2_class_time:.4f}s → {v2_class_result.value:.6f}")

            # Scores should still be identical
            assert abs(legacy_score - v2_class_result.value) < 1e-10

            # Verify types (legacy can return int 0 or float)
            assert isinstance(legacy_score, (int, float))
            assert isinstance(v2_class_result, MetricResult)

    @pytest.mark.asyncio
    async def test_v2_class_no_components_needed(self):
        """Test that V2 class-based RougeScore doesn't require LLM or embeddings."""

        print("\n🔧 Testing V2 RougeScore component requirements:")

        # Should create successfully without any components
        metric = RougeScore(rouge_type="rougeL", mode="fmeasure")

        print(f"   dataclass fields: {list(metric.__dataclass_fields__.keys())}")
        print(f"   has llm field: {'llm' in metric.__dataclass_fields__}")
        print(f"   has embeddings field: {'embeddings' in metric.__dataclass_fields__}")

        # Test that it works
        result = await metric.ascore(
            reference="The capital of France is Paris.",
            response="Paris is the capital of France.",
        )

        print(f"   Score: {result.value:.6f}")

        assert "llm" not in metric.__dataclass_fields__
        assert "embeddings" not in metric.__dataclass_fields__
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0

        print("   ✅ V2 RougeScore works without defining llm/embeddings fields!")

    @pytest.mark.asyncio
    async def test_v2_class_batch_processing(self, sample_data):
        """Test V2 class-based RougeScore batch processing."""

        metric = RougeScore(rouge_type="rougeL", mode="fmeasure")

        # Prepare batch inputs
        batch_inputs = [
            {"reference": case["reference"], "response": case["response"]}
            for case in sample_data[:3]  # Use first 3 cases
        ]

        print(f"\n📦 Testing V2 class batch processing with {len(batch_inputs)} items:")

        # Process batch
        results = await metric.abatch_score(batch_inputs)

        assert len(results) == len(batch_inputs)

        for i, (case, result) in enumerate(zip(sample_data[:3], results)):
            print(f"   Case {i + 1}: {result.value:.6f} - {case['description']}")
            assert isinstance(result.value, float)
            assert 0.0 <= result.value <= 1.0
            assert result.reason is None  # Should be None for successful scoring

        print("   ✅ V2 class batch processing works correctly!")
