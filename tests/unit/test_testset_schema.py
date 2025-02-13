import pytest

from ragas.dataset_schema import (
    EvaluationDataset,
    HumanMessage,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.testset.synthesizers.testset_schema import Testset as RagasTestset
from ragas.testset.synthesizers.testset_schema import (
    TestsetSample as RagasTestsetSample,
)

samples = [
    SingleTurnSample(user_input="What is X", response="Y"),
    MultiTurnSample(
        user_input=[HumanMessage(content="What is X")],
        reference="Y",
    ),
]


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_to_evaluation_dataset(eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    evaluation_dataset = testset.to_evaluation_dataset()
    assert evaluation_dataset == EvaluationDataset(samples=[eval_sample, eval_sample])


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_save_load_csv(tmpdir, eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    testset.to_csv(tmpdir / "csvfile.csv")


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_save_load_jsonl(tmpdir, eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    testset.to_jsonl(tmpdir / "jsonlfile.jsonl")
    loaded_testset = RagasTestset.from_jsonl(tmpdir / "jsonlfile.jsonl")
    assert loaded_testset == testset


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_save_load_hf(tmpdir, eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    hf_testset = testset.to_hf_dataset()
    loaded_testset = RagasTestset.from_hf_dataset(hf_testset)
    assert loaded_testset == testset
