from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample


def test_evaluation_dataset():
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")
    testset_sample = TestsetSample(
        eval_sample=single_turn_sample, synthesizer_name="test"
    )
    testset = Testset(samples=[testset_sample, testset_sample])
    evaluation_dataset = testset.to_evaluation_dataset()
    assert evaluation_dataset == EvaluationDataset(
        samples=[single_turn_sample, single_turn_sample]
    )
