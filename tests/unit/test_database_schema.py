import pytest
from langchain_core.pydantic_v1 import ValidationError

from ragas.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample


def test_evaluation_dataset():
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")

    dataset = EvaluationDataset(samples=[single_turn_sample, single_turn_sample])

    hf_dataset = dataset.to_hf_dataset()

    assert dataset.get_sample_type() == SingleTurnSample
    assert len(hf_dataset) == 2
    assert dataset.features() == {"response", "user_input"}
    assert len(dataset) == 2
    assert dataset[0] == single_turn_sample


def test_single_type_evaluation_dataset():
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")
    multi_turn_sample = MultiTurnSample(
        user_input=[{"content": "What is X"}], response="Y"
    )

    with pytest.raises(ValidationError) as exc_info:
        EvaluationDataset(samples=[single_turn_sample, multi_turn_sample])

    assert "All samples must be of the same type" in str(exc_info.value)
