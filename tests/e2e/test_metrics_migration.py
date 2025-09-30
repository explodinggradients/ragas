"""
End-to-end test for answer_relevancy metric migration using real datasets.

This test file downloads and uses real datasets to test the answer_relevancy metric,
helping verify that it works correctly during migration.
"""

import os
import typing as t
from typing import Any, Dict, List

import pytest
from datasets import Dataset, load_dataset

from ragas import EvaluationDataset, evaluate
from ragas.metrics import answer_relevancy

if t.TYPE_CHECKING:
    pass


def load_wikiqa_dataset_for_ragas(num_samples: int = 5) -> Dataset:
    """
    Load WikiQA dataset and convert to Ragas format.

    This uses the explodinggradients/ragas-wikiqa dataset which is compatible
    with modern HuggingFace datasets (no deprecated scripts).

    Args:
        num_samples: Number of samples to load for testing

    Returns:
        Dataset in Ragas format with keys: user_input, response, retrieved_contexts, reference
    """
    # Load the working dataset (modern format, no deprecated scripts)
    raw_ds = load_dataset(
        "explodinggradients/ragas-wikiqa", split=f"train[:{num_samples}]"
    )

    # Convert to Ragas format - use proper dataset iteration
    converted_data: List[Dict[str, Any]] = []
    for sample in raw_ds:  # type: ignore  # Dataset iteration is valid at runtime
        converted_sample = {
            "user_input": sample["question"],  # type: ignore
            "response": sample["generated_with_rag"],  # type: ignore
            "retrieved_contexts": sample["context"],  # type: ignore
            "reference": sample["correct_answer"],  # type: ignore
        }
        converted_data.append(converted_sample)

    return Dataset.from_list(converted_data)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_answer_relevancy_migration():
    """Test answer_relevancy metric with real dataset."""
    print("\n🧪 Testing answer_relevancy metric migration")

    # Load dataset
    num_samples = 3
    ds = load_wikiqa_dataset_for_ragas(num_samples)

    print("\n📊 Dataset info:")
    print(f"   Total samples: {len(ds)}")
    print(f"   Keys: {list(ds[0].keys())}")

    # Evaluate with answer_relevancy
    print(f"\n🔄 Evaluating {len(ds)} samples with answer_relevancy...")
    result = evaluate(
        EvaluationDataset.from_hf_dataset(ds),
        metrics=[answer_relevancy],
        show_progress=True,
    )

    assert result is not None

    # Show results
    print("\n🎯 Answer Relevancy Results:")
    # Access scores properly from the evaluation result
    if hasattr(result, "scores") and result.scores:
        scores = [score_dict["answer_relevancy"] for score_dict in result.scores]  # type: ignore
        for i, score in enumerate(scores):
            print(f"   Sample {i + 1}: {score:.6f}")

        # Calculate statistics
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        print("\n📈 Score Statistics:")
        print(f"   Average: {avg_score:.6f}")
        print(f"   Min: {min_score:.6f}")
        print(f"   Max: {max_score:.6f}")

        # Verify all scores are valid
        for i, score in enumerate(scores):
            assert 0 <= score <= 1, (
                f"Sample {i + 1} score {score} is not between 0 and 1"
            )

        print(f"✅ All {len(scores)} scores are valid")
    else:
        print("❌ No scores found in result")
        assert False, "Evaluation result has no scores"
