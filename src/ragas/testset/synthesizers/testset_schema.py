from __future__ import annotations

import typing as t

from pydantic import BaseModel

from ragas.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample

if t.TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from pandas import DataFrame as PandasDataframe


class TestsetSample(BaseModel):
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


class Testset(BaseModel):
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

    def to_pandas(self) -> PandasDataframe:
        """
        Converts the Testset to a Pandas DataFrame.
        """
        import pandas as pd

        return pd.DataFrame(self._to_list())

    def to_hf_dataset(self) -> HFDataset:
        """
        Converts the Testset to a Hugging Face Dataset.

        Raises
        ------
        ImportError
            If the 'datasets' library is not installed.
        """
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            raise ImportError(
                "datasets is not installed. Please install it to use this function."
            )

        return HFDataset.from_list(self._to_list())
