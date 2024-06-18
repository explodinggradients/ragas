import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Extractor(ABC):
    @abstractmethod
    def extract(self, text) -> t.Any:
        pass

    @abstractmethod
    def merge_extractors(self, *extractors) -> "Extractor":
        pass


@dataclass
class Regex:
    name: str
    pattern: str

    def __call__(self):
        # Ensure the pattern is a raw string
        if not isinstance(self.pattern, str):
            raise TypeError("Pattern must be a string.")

        if not isinstance(self.name, str):
            raise TypeError("Group name must be a string.")

        # Add the named group to the pattern
        return (
            f"(?P<{self.name}>{self.pattern})"
            if self.name != "merged_extractor"
            else self.pattern
        )
