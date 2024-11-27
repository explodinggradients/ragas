import json
import typing as t

import numpy as np
from pydantic import BaseModel


class PromptAnnotation(BaseModel):
    prompt_input: t.Dict[str, t.Any]
    prompt_output: t.Dict[str, t.Any]
    is_accepted: bool
    edited_output: t.Union[t.Dict[str, t.Any], None]


class SampleAnnotation(BaseModel):
    metric_input: t.Dict[str, t.Any]
    metric_output: float
    prompts: t.Dict[str, PromptAnnotation]
    is_accepted: bool


class MetricAnnotation(BaseModel):

    root: t.Dict[str, t.List[SampleAnnotation]]

    @classmethod
    def from_json(cls, path) -> "MetricAnnotation":

        dataset = json.load(open(path))
        return cls(
            root={
                key: [SampleAnnotation(**sample) for sample in value]
                for key, value in dataset.items()
            }
        )

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: t.Optional[np.random.RandomState] = None,
        stratify: t.Optional[t.List[t.Any]] = None,
    ):
        """
        Split the dataset into training and testing sets.

        Parameters:
            test_size (float): The proportion of the dataset to include in the test split.
            seed (int): Random seed for reproducibility.
            stratify (list): The column values to stratify the split on.
        """
        pass

    def batch(self, batch_size: int, stratiy: t.Optional[str] = None):
        """
        Create a batch iterator.

        Parameters:
            batch_size (int): The number of samples in each batch.
            stratify (str): The column to stratify the batches on.
        """
        pass
