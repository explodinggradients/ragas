"""Represents the types of Notion objects like text, number, select, multi-select, etc."""

__all__ = [
    "T",
    "Field",
    "ID",
    "Text",
    "Title",
    "Select",
    "MultiSelect",
    "URL",
    "NotionFieldMeta",
    "TextNew",
]

import typing as t

from ..exceptions import ValidationError

T = t.TypeVar("T")


class Field(t.Generic[T]):
    """Base class for all Notion field types."""

    NOTION_FIELD_TYPE = ""
    _type: t.Type[T]

    def __init__(self, required: bool = True):
        self.required = required
        self.name: str = ""
        super().__init__()

    def __set_name__(self, owner: t.Type, name: str):
        """Set the field name when the class is created."""
        self.name = name
        if not hasattr(owner, "_fields"):
            owner._fields = {}
        owner._fields[name] = self

    def __get__(self, instance, owner=None):
        """Implement descriptor protocol for getting field values."""
        if instance is None:
            return self
        return instance._values.get(self.name)

    def __set__(self, instance, value):
        """Implement descriptor protocol for setting field values."""
        if instance is None:
            return
        value = self.validate(value)
        instance._values[self.name] = value

    def validate(self, value: t.Any) -> t.Any:
        """Validate the field value."""
        if value is None and self.required:
            raise ValidationError(f"Field {self.name} is required")
        return value

    def _to_notion(self, value: t.Any) -> dict:
        """Convert Python value to Notion format."""
        raise NotImplementedError

    def _from_notion(self, data: dict) -> t.Any:
        """Convert Notion format to Python value."""
        raise NotImplementedError

    def _to_notion_property(self) -> dict:
        """Convert field to Notion property definition format."""
        return {self.name: {"type": self.NOTION_FIELD_TYPE, self.NOTION_FIELD_TYPE: {}}}


class ID(Field[int], int):
    """System ID field type for integer IDs."""

    NOTION_FIELD_TYPE = "unique_id"
    _type = int

    def __new__(cls, *args, **kwargs):
        return int.__new__(cls)

    def __init__(self, required: bool = False):
        super().__init__(required=required)

    def validate(self, value: t.Optional[int]) -> t.Optional[int]:
        value = super().validate(value)
        if value is not None and not isinstance(value, int):
            raise ValidationError(f"ID must be an integer, got {type(value)}")
        return value

    def _to_notion(self, value: int) -> dict:
        return {self.name: {"type": "unique_id", "unique_id": value}}

    def _from_notion(self, data: dict) -> t.Optional[int]:
        if "properties" in data:
            if self.name in data["properties"]:
                return data["properties"][self.name]["unique_id"]["number"]
        else:
            if self.name in data:
                return data[self.name]["unique_id"]["number"]
        # if not found and required, raise error
        if self.required:
            raise ValidationError(
                f"ID field {self.name} is required but not found in the data"
            )
        else:
            return None

    def _to_notion_property(self) -> dict:
        return {self.name: {"type": "unique_id", "unique_id": {"prefix": None}}}


class Text(Field[str], str):
    """Rich text property type."""

    NOTION_FIELD_TYPE = "rich_text"
    _type = str
    CHUNK_SIZE = 2000  # Notion's character limit per rich text block

    def __new__(cls, *args, **kwargs):
        return str.__new__(cls)

    def __init__(self, required: bool = True):
        super().__init__(required=required)

    def _to_notion(self, value: str) -> dict:
        # Split the text into chunks of CHUNK_SIZE characters
        if not value:
            return {self.name: {self.NOTION_FIELD_TYPE: []}}

        chunks = [
            value[i : i + self.CHUNK_SIZE]
            for i in range(0, len(value), self.CHUNK_SIZE)
        ]
        rich_text_array = [{"text": {"content": chunk}} for chunk in chunks]

        return {self.name: {self.NOTION_FIELD_TYPE: rich_text_array}}

    def _from_notion(self, data: dict) -> t.Optional[str]:
        # Handle both direct and properties-wrapped format
        if "properties" in data:
            rich_text = data["properties"][self.name][self.NOTION_FIELD_TYPE]
        else:
            rich_text = data[self.name][self.NOTION_FIELD_TYPE]

        if not rich_text:
            return None

        # Combine all text chunks into a single string
        return "".join(item["text"]["content"] for item in rich_text if "text" in item)


class Title(Field[str], str):
    """Title property type."""

    NOTION_FIELD_TYPE = "title"
    _type = str

    def __new__(cls, *args, **kwargs):
        return str.__new__(cls)

    def __init__(self, required: bool = True):
        super().__init__(required=required)

    def _to_notion(self, value: str) -> dict:
        return {self.name: {self.NOTION_FIELD_TYPE: [{"text": {"content": value}}]}}

    def _from_notion(self, data: dict) -> t.Optional[str]:
        if "properties" in data:
            title = data["properties"][self.name][self.NOTION_FIELD_TYPE]
        else:
            title = data[self.name][self.NOTION_FIELD_TYPE]
        if not title:
            return None
        return title[0]["text"]["content"]


class Select(Field[str], str):
    """Select property type."""

    NOTION_FIELD_TYPE = "select"
    _type = str

    def __new__(cls, *args, **kwargs):
        return str.__new__(cls)

    def __init__(self, options: t.Optional[list[str]] = None, required: bool = True):
        self.options = options
        super().__init__(required=required)

    def validate(self, value: t.Optional[str]) -> t.Optional[str]:
        value = super().validate(value)
        if value == "":  # Allow empty string for optional fields
            return value
        if value is not None and self.options and value not in self.options:
            raise ValidationError(
                f"Value {value} not in allowed options: {self.options}"
            )
        return value

    def _to_notion(self, value: str) -> dict:
        return {self.name: {self.NOTION_FIELD_TYPE: {"name": value}}}

    def _from_notion(self, data: dict) -> t.Optional[str]:
        if "properties" in data:
            select_data = data["properties"][self.name][self.NOTION_FIELD_TYPE]
        else:
            select_data = data[self.name][self.NOTION_FIELD_TYPE]
        if select_data is None:
            return None
        return select_data["name"]

    def _to_notion_property(self) -> dict:
        prop = super()._to_notion_property()
        if self.options:
            prop[self.name]["select"]["options"] = [
                {"name": option} for option in self.options
            ]
        return prop


class MultiSelect(Field[list[str]], list):
    """Multi-select property type."""

    NOTION_FIELD_TYPE = "multi_select"
    _type = list

    def __new__(cls, *args, **kwargs):
        return list.__new__(cls)

    def __init__(self, options: t.Optional[list[str]] = None, required: bool = True):
        self.options = options
        super().__init__(required=required)

    def validate(self, value: t.Optional[list[str]]) -> t.Optional[list[str]]:
        value = super().validate(value)
        if value is not None and self.options:
            invalid_options = [v for v in value if v not in self.options]
            if invalid_options:
                raise ValidationError(
                    f"Values {invalid_options} not in allowed options: {self.options}"
                )
        return value

    def _to_notion(self, value: list[str]) -> dict:
        return {
            self.name: {self.NOTION_FIELD_TYPE: [{"name": option} for option in value]}
        }

    def _from_notion(self, data: dict) -> list[str]:
        if "properties" in data:
            multi_select = data["properties"][self.name][self.NOTION_FIELD_TYPE]
        else:
            multi_select = data[self.name][self.NOTION_FIELD_TYPE]
        if not multi_select:
            return []
        return [item["name"] for item in multi_select]

    def _to_notion_property(self) -> dict:
        prop = super()._to_notion_property()
        if self.options:
            prop[self.name]["multi_select"]["options"] = [
                {"name": option} for option in self.options
            ]
        return prop


class URL(Field[str], str):
    """URL property type."""

    NOTION_FIELD_TYPE = "url"
    _type = str

    def __new__(cls, *args, **kwargs):
        return str.__new__(cls)

    def __init__(self, required: bool = False):
        super().__init__(required=required)

    def validate(self, value: t.Optional[str]) -> t.Optional[str]:
        value = super().validate(value)
        if value is not None and not isinstance(value, str):
            raise ValidationError(f"URL must be a string, got {type(value)}")
        return value

    def _to_notion(self, value: str) -> dict:
        return {self.name: {self.NOTION_FIELD_TYPE: value}}

    def _from_notion(self, data: dict) -> t.Optional[str]:
        if "properties" in data:
            url = data["properties"][self.name][self.NOTION_FIELD_TYPE]
        else:
            url = data[self.name][self.NOTION_FIELD_TYPE]
        return url


T = t.TypeVar("T")


class NotionFieldMeta:
    """Base metadata class for Notion field types."""

    NOTION_FIELD_TYPE: t.ClassVar[str] = ""

    def __init__(self, required: bool = True):
        self.required = required
        self.name: str = ""  # Will be set during model initialization

    def __set_name__(self, owner, name: str):
        """Set field name when used directly as class attribute."""
        self.name = name

    def validate(self, value: t.Any) -> t.Any:
        """Validate field value."""
        if value is None and self.required:
            raise ValueError(f"Field {self.name} is required")
        return value

    def to_notion(self, value: t.Any) -> dict:
        """Convert Python value to Notion format."""
        raise NotImplementedError()

    def from_notion(self, data: dict) -> t.Any:
        """Convert Notion format to Python value."""
        raise NotImplementedError()

    def to_notion_property(self) -> dict:
        """Convert field to Notion property definition."""
        return {self.name: {"type": self.NOTION_FIELD_TYPE, self.NOTION_FIELD_TYPE: {}}}


class TextNew(NotionFieldMeta):
    """Rich text property type for Notion."""

    NOTION_FIELD_TYPE = "rich_text"
    CHUNK_SIZE = 2000  # Notion's character limit per rich text block

    def __init__(self, required: bool = True):
        super().__init__(required=required)

    def to_notion(self, value: str) -> dict:
        # Split text into chunks of CHUNK_SIZE characters
        if not value:
            return {self.name: {self.NOTION_FIELD_TYPE: []}}

        chunks = [
            value[i : i + self.CHUNK_SIZE]
            for i in range(0, len(value), self.CHUNK_SIZE)
        ]
        rich_text_array = [{"text": {"content": chunk}} for chunk in chunks]

        return {self.name: {self.NOTION_FIELD_TYPE: rich_text_array}}

    def from_notion(self, data: dict) -> t.Optional[str]:
        # Handle both direct and properties-wrapped format
        if "properties" in data:
            if self.name in data["properties"]:
                rich_text = data["properties"][self.name][self.NOTION_FIELD_TYPE]
            else:
                return None
        else:
            if self.name in data:
                rich_text = data[self.name][self.NOTION_FIELD_TYPE]
            else:
                return None

        if not rich_text:
            return None

        # Combine all text chunks into a single string
        return "".join(item["text"]["content"] for item in rich_text if "text" in item)
