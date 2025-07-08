"""Experiments hold the results of an experiment against a dataset."""

__all__ = ["Experiment"]

import typing as t

from pydantic import BaseModel

from .dataset import DataTable


class Experiment(DataTable):
    def __init__(
        self,
        name: str,
        model: t.Type[BaseModel],
        experiment_id: str,
        backend,  # DataTableBackend instance
    ):
        self.experiment_id = experiment_id
        super().__init__(
            name=name,
            data_model=model,
            dataset_id=experiment_id,
            datatable_type="experiments",
            backend=backend,
        )

    def __str__(self):
        return f"Experiment(name={self.name}, model={self.model.__name__}, len={len(self._entries)})"

    __repr__ = __str__
