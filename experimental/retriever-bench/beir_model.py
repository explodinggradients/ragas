from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import numpy as np

if t.TYPE_CHECKING:
    from torch import Tensor


class BeirRetriever(ABC):
    @abstractmethod
    def encode_queries(
        self, queries: list[str], batch_size: int = 16, **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        ...

    @abstractmethod
    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list],
        batch_size: int = 8,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor:
        ...


class BeirCrossEncoder(ABC):
    @abstractmethod
    def predict(
        self,
        sentences: list[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> list[float]:
        ...
