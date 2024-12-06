import typing as t

import pytest

from ragas.dataset_schema import (
    EvaluationDataset,
    HumanMessage,
    MultiTurnSample,
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
    SingleTurnSample,
)

samples = [
    SingleTurnSample(user_input="What is X", response="Y"),
    MultiTurnSample(
        user_input=[HumanMessage(content="What is X")],
        reference="Y",
    ),
]


def create_sample_annotation(metric_output):
    return SampleAnnotation(
        metric_input={
            "response": "",
            "reference": "",
            "user_input": "",
        },
        metric_output=metric_output,
        prompts={
            "single_turn_aspect_critic_prompt": PromptAnnotation(
                prompt_input={
                    "response": "",
                    "reference": "",
                    "user_input": "",
                },
                prompt_output={"reason": "", "verdict": 1},
                is_accepted=True,
                edited_output=None,
            )
        },
        is_accepted=True,
        target=None,
    )


def test_loader_sample():

    annotated_samples = [create_sample_annotation(1) for _ in range(10)] + [
        create_sample_annotation(0) for _ in range(10)
    ]
    test_dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)
    sample = test_dataset.sample(2)
    assert len(sample) == 2

    sample = test_dataset.sample(2, stratify_key="metric_output")
    assert len(sample) == 2
    assert sum(item["metric_output"] for item in sample) == 1


def test_loader_batch():

    annotated_samples = [create_sample_annotation(1) for _ in range(10)] + [
        create_sample_annotation(0) for _ in range(10)
    ]
    dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)
    batches = dataset.batch(batch_size=2)
    assert all([len(item) == 2 for item in batches])

    batches = dataset.stratified_batches(batch_size=2, stratify_key="metric_output")
    assert all(sum([item["metric_output"] for item in batch]) == 1 for batch in batches)


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset(eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    hf_dataset = dataset.to_hf_dataset()

    assert dataset.get_sample_type() is type(eval_sample)
    assert len(hf_dataset) == 2
    assert len(dataset) == 2
    assert dataset[0] == eval_sample

    dataset_from_hf = EvaluationDataset.from_hf_dataset(hf_dataset)
    assert dataset_from_hf == dataset


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset_save_load_csv(tmpdir, eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    # save and load to csv
    csv_path = tmpdir / "csvfile.csv"
    dataset.to_csv(csv_path)


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset_save_load_jsonl(tmpdir, eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    # save and load to jsonl
    jsonl_path = tmpdir / "jsonlfile.jsonl"
    dataset.to_jsonl(jsonl_path)
    loaded_dataset = EvaluationDataset.from_jsonl(jsonl_path)
    assert loaded_dataset == dataset


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset_load_from_hf(eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    # convert to and load from hf dataset
    hf_dataset = dataset.to_hf_dataset()
    loaded_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)
    assert loaded_dataset == dataset


@pytest.mark.parametrize("eval_sample", samples)
def test_single_type_evaluation_dataset(eval_sample):
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")
    multi_turn_sample = MultiTurnSample(
        user_input=[{"content": "What is X"}],
        response="Y",  # type: ignore (this type error is what we want to test)
    )

    with pytest.raises(ValueError) as exc_info:
        EvaluationDataset(samples=[single_turn_sample, multi_turn_sample])

    assert "All samples must be of the same type" in str(exc_info.value)


def test_base_eval_sample():
    from ragas.dataset_schema import BaseSample

    class FakeSample(BaseSample):
        user_input: str
        response: str
        reference: t.Optional[str] = None

    fake_sample = FakeSample(user_input="What is X", response="Y")
    assert fake_sample.to_dict() == {"user_input": "What is X", "response": "Y"}
    assert fake_sample.get_features() == ["user_input", "response"]


def test_evaluation_dataset_iter():
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")

    dataset = EvaluationDataset(samples=[single_turn_sample, single_turn_sample])

    for sample in dataset:
        assert sample == single_turn_sample


def test_evaluation_dataset_type():
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")
    multi_turn_sample = MultiTurnSample(
        user_input=[{"content": "What is X"}],
        response="Y",  # type: ignore (this type error is what we want to test)
    )

    dataset = EvaluationDataset(samples=[single_turn_sample])
    assert dataset.get_sample_type() == SingleTurnSample

    dataset = EvaluationDataset(samples=[multi_turn_sample])
    assert dataset.get_sample_type() == MultiTurnSample
