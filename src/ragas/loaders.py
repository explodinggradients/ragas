import json
import random
import typing as t
from collections import defaultdict

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
        return SingleMetricAnnotation(name=key, samples=self.root[key])

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

    def __len__(self):
        return sum(len(value) for value in self.root.values())


class SingleMetricAnnotation(BaseModel):
    name: str
    samples: t.List[SampleAnnotation]

    def __getitem__(self, key):
        return getattr(self, key)

    @classmethod
    def from_json(cls, path) -> "SingleMetricAnnotation":

        dataset = json.load(open(path))

        return cls(
            name=dataset["name"],
            samples=[SampleAnnotation(**sample) for sample in dataset["samples"]],
        )

    def filter(self, function: t.Optional[t.Callable] = None):

        if function is None:
            function = lambda x: True  # noqa: E731

        return SingleMetricAnnotation(
            name=self.name,
            samples=[sample for sample in self.samples if function(sample)],
        )

    def __len__(self):
        return len(self.samples)

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
        drop_last_batch: bool = False,
    ):
        """
        Create a batch iterator.

        Parameters:
            batch_size (int): The number of samples in each batch.
            stratify (str): The column to stratify the batches on.
            drop_last_batch (bool): Whether to drop the last batch if it is smaller than the specified batch size.
        """

        samples = self.samples[:]
        random.shuffle(samples)

        all_batches = [
            samples[i : i + batch_size]
            for i in range(0, len(samples), batch_size)
            if len(samples[i : i + batch_size]) == batch_size or not drop_last_batch
        ]

        return all_batches

    def stratified_batches(
        self,
        batch_size: int,
        stratify_key: str,
        drop_last_batch: bool = False,
        replace: bool = False,
    ) -> t.List[t.List[SampleAnnotation]]:
        """
        Create stratified batches based on a specified key, ensuring proportional representation.

        Parameters:
            batch_size (int): Number of samples per batch.
            stratify_key (str): Key in `metric_input` used for stratification (e.g., class labels).
            drop_last_batch (bool): If True, drops the last batch if it has fewer samples than `batch_size`.
            replace (bool): If True, allows reusing samples from the same class to fill a batch if necessary.

        Returns:
            List[List[SampleAnnotation]]: A list of stratified batches, each batch being a list of SampleAnnotation objects.
        """
        # Group samples based on the stratification key
        class_groups = defaultdict(list)
        for sample in self.samples:
            key = sample.metric_input.get(stratify_key)
            if key is None:
                raise ValueError(
                    f"Stratify key '{stratify_key}' not found in metric_input."
                )
            class_groups[key].append(sample)

        # Shuffle each class group for randomness
        for group in class_groups.values():
            random.shuffle(group)

        # Determine the number of batches required
        total_samples = len(self.samples)
        num_batches = (
            total_samples // batch_size
            if drop_last_batch
            else (total_samples + batch_size - 1) // batch_size
        )
        samples_per_class_per_batch = {
            cls: max(1, len(samples) // num_batches)
            for cls, samples in class_groups.items()
        }

        # Create stratified batches
        all_batches = []
        while len(all_batches) < num_batches:
            batch = []
            for cls, samples in list(class_groups.items()):
                # Determine the number of samples to take from this class
                count = min(
                    samples_per_class_per_batch[cls],
                    len(samples),
                    batch_size - len(batch),
                )
                if count > 0:
                    # Add samples from the current class
                    batch.extend(samples[:count])
                    class_groups[cls] = samples[count:]  # Remove used samples
                elif replace and len(batch) < batch_size:
                    # Reuse samples if `replace` is True
                    batch.extend(random.choices(samples, k=batch_size - len(batch)))

            # Shuffle the batch to mix classes
            random.shuffle(batch)
            if len(batch) == batch_size or not drop_last_batch:
                all_batches.append(batch)

        return all_batches
