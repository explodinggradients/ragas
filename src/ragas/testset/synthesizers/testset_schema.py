from __future__ import annotations

import typing as t

from pydantic import BaseModel

from ragas.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample

if t.TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from pandas import DataFrame as PandasDataframe


class TestsetSample(BaseModel):
    eval_sample: t.Union[SingleTurnSample, MultiTurnSample]
    synthesizer_name: str


class Testset(BaseModel):
    samples: t.List[TestsetSample]

    def to_evaluation_dataset(self) -> EvaluationDataset:
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
        import pandas as pd

        return pd.DataFrame(self._to_list())

    def to_hf_dataset(self) -> HFDataset:
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            raise ImportError(
                "datasets is not installed. Please install it to use this function."
            )

        return HFDataset.from_list(self._to_list())
