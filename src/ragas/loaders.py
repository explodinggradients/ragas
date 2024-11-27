import json
import typing as t

import numpy as np
from pydantic import BaseModel


class PromptAnnotation(BaseModel):
    prompt_input: t.Dict[str, t.Any]
    prompt_output: t.Dict[str, t.Any]
    is_accepted: bool
    edited_output: t.Union[t.Dict[str, t.Any], None]

    def __getitem__(self, key):
        return getattr(self, key)


class SampleAnnotation(BaseModel):
    metric_input: t.Dict[str, t.Any]
    metric_output: float
    prompts: t.Dict[str, PromptAnnotation]
    is_accepted: bool

    def __getitem__(self, key):
        return getattr(self, key)


class MetricAnnotation(BaseModel):

    root: t.Dict[str, t.List[SampleAnnotation]]

    def __getitem__(self, key):
        return self.root[key]

    @classmethod
    def from_json(cls, path, metric_name: t.Optional[str]) -> "MetricAnnotation":

        dataset = json.load(open(path))
        if metric_name is not None and metric_name not in dataset:
            raise ValueError(f"Split {metric_name} not found in the dataset.")

        return cls(
            root={
                key: [SampleAnnotation(**sample) for sample in value]
                for key, value in dataset.items()
                if metric_name is None or key == metric_name
            }
        )

    def filter(self, function: t.Optional[t.Callable] = None):

        if function is None:
            function = lambda x: True  # noqa: E731

        return MetricAnnotation(
            root={
                key: [sample for sample in value if function(sample)]
                for key, value in self.root.items()
            }
        )

    def __len__(self):
        return sum(len(value) for value in self.root.values())

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

    def batch(
        self,
        batch_size: int,
        stratify: t.Optional[str] = None,
        drop_last_batch: bool = False,
    ):
        """
        Create a batch iterator.

        Parameters:
            batch_size (int): The number of samples in each batch.
            stratify (str): The column to stratify the batches on.
            drop_last_batch (bool): Whether to drop the last batch if it is smaller than the specified batch size.
        """
