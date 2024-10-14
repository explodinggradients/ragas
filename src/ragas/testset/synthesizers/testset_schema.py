from __future__ import annotations

import typing as t

from ragas.dataset_schema import BaseSample, RagasDataset

if t.TYPE_CHECKING:
    from ragas.dataset_schema import (
        EvaluationDataset,
        MultiTurnSample,
        SingleTurnSample,
    )


class TestsetSample(BaseSample):
    """
    Represents a sample in a test set.

    Attributes
    ----------
    eval_sample : Union[SingleTurnSample, MultiTurnSample]
        The evaluation sample, which can be either a single-turn or multi-turn sample.
    synthesizer_name : str
        The name of the synthesizer used to generate this sample.
    """

    eval_sample: t.Union[SingleTurnSample, MultiTurnSample]
    synthesizer_name: str


class Testset(RagasDataset[TestsetSample]):
    """
    Represents a test set containing multiple test samples.

    Attributes
    ----------
    samples : List[TestsetSample]
        A list of TestsetSample objects representing the samples in the test set.
    """

    samples: t.List[TestsetSample]

    def to_evaluation_dataset(self) -> EvaluationDataset:
        """
        Converts the Testset to an EvaluationDataset.
        """
        return EvaluationDataset(
            samples=[sample.eval_sample for sample in self.samples]
        )

    def _to_list(self) -> t.List[t.Dict]:
        eval_list = self.to_evaluation_dataset()._to_list()
        testset_list_without_eval_sample = [
            sample.model_dump(exclude={"eval_sample"}) for sample in self.samples
        ]
        testset_list = [
            {**eval_sample, **sample}
            for eval_sample, sample in zip(eval_list, testset_list_without_eval_sample)
        ]
        return testset_list
