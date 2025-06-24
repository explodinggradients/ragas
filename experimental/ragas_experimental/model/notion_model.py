"""NotionModel is a class that allows you to create a model of a Notion database."""

__all__ = ["NotionModelMeta", "NotionModel"]

import typing as t
from datetime import datetime

from fastcore.utils import patch, patch_to

from ..exceptions import ValidationError
from .notion_typing import ID, Field


class NotionModelMeta(type):
    """Metaclass for NotionModel to handle field registration."""

    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        _fields: t.Dict[str, Field] = {}

        # Collect fields from base classes
        for base in bases:
            if hasattr(base, "_fields"):
                _fields.update(base._fields)

        # Collect fields from class variables and type annotations
        for key, value in namespace.items():
            # Skip internal attributes
            if key.startswith("_"):
                continue

            # Handle field instances directly defined in class
            if isinstance(value, Field):
                _fields[key] = value
            # Handle annotated but not instantiated fields
            elif (
                key in namespace.get("__annotations__", {})
                and isinstance(value, type)
                and issubclass(value, Field)
            ):
                _fields[key] = value()

        namespace["_fields"] = _fields
        return super().__new__(mcs, name, bases, namespace)


class NotionModel(metaclass=NotionModelMeta):
    """Base class for Notion database models.

    Represents a row in a Notion database with typed fields that map to
    Notion property values.
    """

    _fields: t.ClassVar[t.Dict[str, Field]]
    _created_time: t.Optional[datetime] = None
    _last_edited_time: t.Optional[datetime] = None
    _page_id: t.Optional[str] = None

    def __init__(self, **kwargs):
        self._values: t.Dict[str, t.Any] = {}
        self._page_id = kwargs.pop("page_id", None)  # Extract page_id from kwargs
        self._created_time = kwargs.pop("created_time", None)
        self._last_edited_time = kwargs.pop("last_edited_time", None)

        # Get required fields
        required_fields = {
            name
            for name, field in self._fields.items()
            if field.required and name not in kwargs
        }

        if required_fields:
            raise ValidationError(f"Missing required fields: {required_fields}")

        # Set values and validate
        for name, value in kwargs.items():
            if name in self._fields:
                setattr(self, name, value)
            else:
                raise ValidationError(f"Unknown field: {name}")

    def __setattr__(self, name: str, value: t.Any):
        """Handle field validation on attribute setting."""
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        field = self._fields.get(name)
        if field is not None:
            value = field.validate(value)
            self._values[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> t.Any:
        """Handle field access."""
        if name in self._values:
            return self._values[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        """Return a string representation of the model instance."""
        class_name = self.__class__.__name__
        parts = []

        # First add ID fields
        for name, field in self.__class__._fields.items():
            if isinstance(field, ID) and name in self._values:
                value = self._values[name]
                if value is not None:
                    parts.append(f"{name}={repr(value)}")

        # Then add other fields in declaration order
        for name, field in self.__class__._fields.items():
            if not isinstance(field, ID) and name in self._values:
                value = self._values[name]
                if value is not None:
                    if isinstance(value, str):
                        parts.append(f"{name}='{value}'")
                    else:
                        parts.append(f"{name}={repr(value)}")

        return f"{class_name}({' '.join(parts)})"


@patch
def to_notion(self: NotionModel) -> dict:
    """Convert the model to Notion API format."""
    properties = {}
    for name, field in self._fields.items():
        if name in self._values:
            value = self._values[name]
            if value is not None:
                properties.update(field._to_notion(value))
    return {"properties": properties}


@patch_to(NotionModel, cls_method=True)
def from_notion(cls, data: dict) -> "NotionModel":
    """Create a model instance from Notion API data."""
    values = {}
    for name, field in cls._fields.items():
        if name in data.get("properties", {}):
            values[name] = field._from_notion({"properties": data["properties"]})

    # Handle system properties
    if "id" in data:
        values["page_id"] = data["id"]  # Set page_id from Notion's id
    if "created_time" in data:
        values["created_time"] = datetime.fromisoformat(
            data["created_time"].replace("Z", "+00:00")
        )
    if "last_edited_time" in data:
        values["last_edited_time"] = datetime.fromisoformat(
            data["last_edited_time"].replace("Z", "+00:00")
        )

    return cls(**values)
