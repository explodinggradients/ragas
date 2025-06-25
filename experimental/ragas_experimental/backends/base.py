"""Base classes for dataset backends."""

from abc import ABC, abstractmethod
import typing as t


class DatasetBackend(ABC):
    """Abstract base class for dataset backends.

    All dataset storage backends must implement these methods.
    """

    @abstractmethod
    def initialize(self, dataset):
        """Initialize the backend with dataset information"""
        pass

    @abstractmethod
    def get_column_mapping(self, model):
        """Get mapping between model fields and backend columns"""
        pass

    @abstractmethod
    def load_entries(self, model_class):
        """Load all entries from storage"""
        pass

    @abstractmethod
    def append_entry(self, entry):
        """Add a new entry to storage and return its ID"""
        pass

    @abstractmethod
    def update_entry(self, entry):
        """Update an existing entry in storage"""
        pass

    @abstractmethod
    def delete_entry(self, entry_id):
        """Delete an entry from storage"""
        pass

    @abstractmethod
    def get_entry_by_field(self, field_name: str, field_value: t.Any, model_class):
        """Get an entry by field value"""
        pass
