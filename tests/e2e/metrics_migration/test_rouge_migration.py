"""E2E tests for ROUGE score metric migration from v1 (class-based) to v2 (decorator-based)."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import MetricResult, RougeScore
from ragas.metrics.v2 import rouge_score


class TestRougeE2EMigration:
    """E2E test compatibility between legacy RougeScore class and new rouge_score decorator."""

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
    async def test_legacy_rouge_score_vs_v2_rouge_score_e2e_compatibility(
        self, sample_data, rouge_type, mode
    ):
        """E2E test that legacy and v2 implementations produce identical scores across various text pairs."""

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing {rouge_type} {mode} - Case {i + 1}: {data['description']}"
            )
            print(f"   Reference: {data['reference'][:50]}...")
            print(f"   Response:  {data['response'][:50]}...")

            # Legacy v1
            legacy_rouge_score = RougeScore(rouge_type=rouge_type, mode=mode)
            legacy_sample = SingleTurnSample(
                user_input="dummy",
                response=data["response"],
                reference=data["reference"],
            )
            legacy_score = await legacy_rouge_score._single_turn_ascore(
                legacy_sample, None
            )

            # V2
            v2_rouge_score_result = await rouge_score.ascore(
                reference=data["reference"],
                response=data["response"],
                rouge_type=rouge_type,
                mode=mode,
            )

            # Verify exact match
            score_diff = abs(legacy_score - v2_rouge_score_result.value)
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_rouge_score_result.value:.6f}")
            print(f"   Diff:   {score_diff:.10f}")

            assert score_diff < 1e-10, (
                f"Case {i + 1} ({data['description']}): {rouge_type} {mode} mismatch: {legacy_score} != {v2_rouge_score_result.value}"
            )

            # Verify types (legacy can return int 0 or float)
            assert isinstance(legacy_score, (int, float))
            assert isinstance(v2_rouge_score_result, MetricResult)

            print("   ✅ Identical scores!")

    @pytest.mark.asyncio
    async def test_rouge_score_performance_comparison(self, sample_data):
        """Compare performance characteristics between legacy and v2."""
        import time

        # Test with multiple configurations
        configs = [("rouge1", "fmeasure"), ("rougeL", "fmeasure")]
        test_case = sample_data[0]  # Use first realistic test case

        for rouge_type, mode in configs:
            print(f"\n⚡ Performance test: {rouge_type} {mode}")

            # Legacy timing
            legacy_rouge_score = RougeScore(rouge_type=rouge_type, mode=mode)
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

            # V2 timing
            start_time = time.time()
            v2_result = await rouge_score.ascore(
                reference=test_case["reference"],
                response=test_case["response"],
                rouge_type=rouge_type,
                mode=mode,
            )
            v2_time = time.time() - start_time

            print(f"   Legacy: {legacy_time:.4f}s → {legacy_score:.6f}")
            print(f"   V2:     {v2_time:.4f}s → {v2_result.value:.6f}")
            print(
                f"   Speedup: {legacy_time / v2_time:.2f}x"
                if v2_time > 0
                else "   V2 faster"
            )

            # Scores should still be identical
            assert abs(legacy_score - v2_result.value) < 1e-10
            # Verify types (legacy can return int 0 or float)
            assert isinstance(legacy_score, (int, float))
            assert isinstance(v2_result, MetricResult)
