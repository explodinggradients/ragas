"""Experiments hold the results of an experiment against a dataset."""

__all__ = ["Experiment"]

import typing as t

from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

from .backends.ragas_api_client import RagasApiClient
from .dataset import Dataset


class Experiment(Dataset):
    def __init__(
        self,
        name: str,
        model: t.Type[BaseModel],
        project_id: str,
        experiment_id: str,
        ragas_api_client: t.Optional[RagasApiClient] = None,
        backend: t.Literal["ragas/app", "local/csv"] = "ragas/app",
        local_root_dir: t.Optional[str] = None,
    ):
        self.experiment_id = experiment_id
        super().__init__(
            name=name,
            model=model,
            project_id=project_id,
            dataset_id=experiment_id,
            ragas_api_client=ragas_api_client,
            backend=backend,
            local_root_dir=local_root_dir,
            datatable_type="experiments",
        )

    def __str__(self):
        return f"Experiment(name={self.name}, model={self.model.__name__}, len={len(self._entries)})"

    __repr__ = __str__
