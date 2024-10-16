from __future__ import annotations

import typing as t

from ragas.dataset_schema import (
    BaseSample,
    EvaluationDataset,
    MultiTurnSample,
    RagasDataset,
    SingleTurnSample,
)

if t.TYPE_CHECKING:
    from ragas.dataset_schema import MultiTurnSample, SingleTurnSample


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

    def to_list(self) -> t.List[t.Dict]:
        """
        Converts the Testset to a list of dictionaries.
        """
        return [sample.model_dump() for sample in self.samples]

    @classmethod
    def from_list(cls, data: t.List[t.Dict]) -> Testset:
        """
        Converts a list of dictionaries to a Testset.
        """
        # first create the samples
        samples = []
        for sample in data:
            eval_sample = sample["eval_sample"]

            # if user_input is a list it is MultiTurnSample
            if "user_input" in eval_sample and not isinstance(
                eval_sample.get("user_input"), list
            ):
                eval_sample = SingleTurnSample(**sample["eval_sample"])
            else:
                eval_sample = MultiTurnSample(**sample["eval_sample"])

            samples.append(
                TestsetSample(
                    eval_sample=eval_sample, synthesizer_name=sample["synthesizer_name"]
                )
            )
        # then create the testset
        return Testset(samples=samples)
