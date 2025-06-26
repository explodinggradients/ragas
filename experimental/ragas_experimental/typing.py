"""Field Metadata for python's `t.Annotate`."""

__all__ = [
    "SUPPORTED_BACKENDS",
    "DEFAULT_COLUMN_SETTINGS",
    "COLOR_MAP",
    "ColumnType",
    "FieldMeta",
    "Number",
    "Text",
    "Url",
    "get_colors_for_options",
    "Select",
    "MultiSelect",
    "Checkbox",
    "Date",
    "Custom",
    "ModelConverter",
    "infer_metric_result_type",
    "infer_field_type",
]

import typing as t
from datetime import date, datetime
from enum import Enum

from fastcore.utils import patch

from .metric.result import MetricResult

# Define supported backends
SUPPORTED_BACKENDS = t.Literal["local/csv", "ragas/app", "box/csv"]


class ColumnType(str, Enum):
    """Column types supported by the Ragas API."""

    NUMBER = "number"
    TEXT = "longText"
    SELECT = "select"
    MULTI_SELECT = "multiSelect"
    CHECKBOX = "checkbox"
    DATE = "date"
    URL = "url"
    CUSTOM = "custom"


DEFAULT_COLUMN_SETTINGS = {
    "width": 255,
    "isVisible": True,
    "isEditable": True,
}


class FieldMeta:
    """Base metadata for field type annotations."""

    def __init__(
        self,
        type,
        required=True,
        id: t.Optional[str] = None,
        settings: t.Optional[dict] = None,
    ):
        self.type = type
        self.required = required
        self.id = id
        self.settings = DEFAULT_COLUMN_SETTINGS.copy()
        # if settings is provided, update the settings
        if settings:
            self.settings.update(settings)


class Number(FieldMeta):
    """Number field metadata."""

    def __init__(
        self,
        min_value: t.Optional[float] = None,
        max_value: t.Optional[float] = None,
        required: bool = True,
        id: t.Optional[str] = None,
    ):
        settings = {}
        if min_value is not None or max_value is not None:
            settings["range"] = {}
            if min_value is not None:
                settings["range"]["min"] = min_value
            if max_value is not None:
                settings["range"]["max"] = max_value
        super().__init__(ColumnType.NUMBER, required, id, settings=settings)


class Text(FieldMeta):
    """Text field metadata."""

    def __init__(
        self, max_length: int = 1000, required: bool = True, id: t.Optional[str] = None
    ):
        settings = {}
        if max_length is not None:
            settings["max_length"] = max_length
        super().__init__(ColumnType.TEXT, required, id, settings=settings)


class Url(FieldMeta):
    """Url field metadata."""

    def __init__(self, required: bool = True, id: t.Optional[str] = None):
        settings = {}
        super().__init__(ColumnType.URL, required, id, settings=settings)


# dict of possible colors for select fields
COLOR_MAP = {
    "red": "hsl(0, 85%, 60%)",
    "orange": "hsl(30, 85%, 60%)",
    "amber": "hsl(45, 85%, 60%)",
    "yellow": "hsl(60, 85%, 60%)",
    "lime": "hsl(90, 85%, 60%)",
    "green": "hsl(120, 85%, 60%)",
    "teal": "hsl(160, 85%, 60%)",
    "cyan": "hsl(180, 85%, 60%)",
    "sky": "hsl(200, 85%, 60%)",
    "blue": "hsl(210, 85%, 60%)",
    "indigo": "hsl(240, 85%, 60%)",
    "violet": "hsl(270, 85%, 60%)",
    "purple": "hsl(280, 85%, 60%)",
    "fuchsia": "hsl(300, 85%, 60%)",
    "pink": "hsl(330, 85%, 60%)",
}


def get_colors_for_options(options, color_names=None):
    """
    Assign colors to options from the COLOR_MAP.

    Args:
        options: List of option names
        color_names: Optional list of specific color names to use from COLOR_MAP
                    If None, colors will be assigned in order from COLOR_MAP

    Returns:
        List of option objects with name, value, and color properties
    """
    if color_names is None:
        # Use colors in order from COLOR_MAP (cycling if needed)
        available_colors = list(COLOR_MAP.values())
        color_values = [
            available_colors[i % len(available_colors)] for i in range(len(options))
        ]
    else:
        # Use specified colors
        color_values = [
            COLOR_MAP.get(color, COLOR_MAP["blue"]) for color in color_names
        ]
        # If fewer colors than options, cycle the colors
        if len(color_values) < len(options):
            color_values = [
                color_values[i % len(color_values)] for i in range(len(options))
            ]

    return [
        {"name": option, "value": option, "color": color_values[i]}
        for i, option in enumerate(options)
    ]


class Select(FieldMeta):
    """Select field metadata."""

    def __init__(
        self,
        options: t.Optional[t.List[str]] = None,
        required: bool = True,
        colors: t.Optional[t.List[str]] = None,
    ):
        settings = {}

        # store the colors for later use when combining with Literal types
        self.colors = colors

        if options:
            if colors:
                settings["options"] = get_colors_for_options(options, colors)
            else:
                settings["options"] = get_colors_for_options(options)
        super().__init__(ColumnType.SELECT, required, settings=settings)


class MultiSelect(FieldMeta):
    """MultiSelect field metadata."""

    def __init__(self, options: t.Optional[t.List[str]] = None, required: bool = True):
        settings = {}
        if options:
            settings["options"] = [{"name": option} for option in options]
        super().__init__(ColumnType.MULTI_SELECT, required, settings=settings)


class Checkbox(FieldMeta):
    """Checkbox field metadata."""

    def __init__(self, required: bool = True):
        super().__init__(ColumnType.CHECKBOX, required)


class Date(FieldMeta):
    """Date field metadata."""

    def __init__(self, include_time: bool = False, required: bool = True):
        settings = {}
        if include_time:
            settings["include_time"] = include_time
        super().__init__(ColumnType.DATE, required, settings=settings)


class Custom(FieldMeta):
    """Custom field metadata."""

    def __init__(self, custom_type: str = "", required: bool = True):
        settings = {}
        if custom_type:
            settings["type"] = custom_type
        super().__init__(ColumnType.CUSTOM, required, settings=settings)


class ModelConverter:
    """Convert Pydantic models to Ragas API columns and rows."""


def infer_metric_result_type(field_value):
    """Infer field type from a MetricResult instance."""
    if field_value is None:
        return Text()

    # Infer type based on the _result type
    result_value = field_value._result

    if isinstance(result_value, (int, float)):
        return Number()
    elif isinstance(result_value, bool):
        return Checkbox()
    elif isinstance(result_value, (list, tuple)):
        # For ranking metrics that return lists
        return Text()
    else:
        # Default to Text for string or other types
        return Text()


def infer_field_type(annotation, field_info):
    """Infer field type from Python type annotation."""
    # Check for Annotated with our custom metadata
    origin = t.get_origin(annotation)
    args = t.get_args(annotation)

    # Check if this is a MetricResult type
    if annotation is MetricResult or (
        hasattr(annotation, "__origin__") and annotation.__origin__ is MetricResult
    ):
        # Default to Text since we can't determine the result type statically
        return Text()

    # If this is an Annotated field then it will have metadata
    if field_info.metadata:
        # Check if we have Select field metadata and base type is Literal
        field_meta = None
        for arg in field_info.metadata:
            if isinstance(arg, FieldMeta):
                field_meta = arg
                break

        if field_meta is not None:
            # if it's a URL field, return it
            if isinstance(field_meta, Url):
                return field_meta

            if isinstance(field_meta, Select) and origin is t.Literal:
                # Special handling for Literal types with Select metadata
                literal_values = list(args)

                # If Select has colors but no options, use the literal values as options
                if (
                    not field_meta.settings.get("options")
                    and "colors" in field_meta.__dict__
                ):
                    colors = field_meta.__dict__["colors"]
                    return Select(options=literal_values, colors=colors)

                # If no colors specified, just use literal values as options
                if not field_meta.settings.get("options"):
                    return Select(options=literal_values)

            # for any other field metadata, just return the field metadata
            return field_meta

        # If no field metadata found, infer from the base type
        return infer_field_type(args[0], field_info)

    # Handle Optional, List, etc.
    if origin is t.Union:
        if type(None) in args:
            # This is Optional[T]
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                # Get the field type of the non-None arg
                field_meta = infer_field_type(non_none_args[0], field_info)
                field_meta.required = False
                return field_meta

    # Handle List and array types
    # NOTE: here we are converting lists to strings, except for literal types
    if origin is list or origin is t.List:
        if len(args) > 0:
            # Check if it's a list of literals
            if t.get_origin(args[0]) is t.Literal:
                literal_options = t.get_args(args[0])
                return MultiSelect(options=list(literal_options))
            # Otherwise just a regular list
            return Text()  # Default to Text for lists

    # Handle Literal
    if origin is t.Literal:
        return Select(options=list(args))

    # Basic type handling
    if annotation is str:
        return Text()
    elif annotation is int or annotation is float:
        return Number()
    elif annotation is bool:
        return Checkbox()
    elif annotation is datetime or annotation is date:
        return Date(include_time=annotation is datetime)

    # Default to Text for complex or unknown types
    return Text()


@patch(cls_method=True)
def model_to_columns(cls: ModelConverter, model_class):
    """Convert a Pydantic model class to Ragas API column definitions."""
    columns = []
    for field_name, field_info in model_class.model_fields.items():
        # Get the field's type annotation
        annotation = field_info.annotation

        # Special handling for MetricResult fields
        if (
            annotation is MetricResult
            or (
                hasattr(annotation, "__origin__")
                and annotation.__origin__ is MetricResult
            )
            or (
                hasattr(field_info, "annotation")
                and str(field_info.annotation).find("MetricResult") != -1
            )
        ):

            # Create column for the result value
            field_meta = infer_field_type(annotation, field_info)
            column = {
                "id": field_name,
                "name": field_name,
                "type": field_meta.type.value,
                "settings": field_meta.settings.copy(),
            }
            columns.append(column)

            # Create additional column for the reason
            reason_column = {
                "id": f"{field_name}_reason",
                "name": f"{field_name}_reason",
                "type": ColumnType.TEXT.value,
                "settings": Text().settings.copy(),
                "editable": True,
            }
            columns.append(reason_column)
        else:
            # Regular field handling
            field_meta = infer_field_type(annotation, field_info)

            column = {
                "id": field_name,
                "name": field_name,
                "type": field_meta.type.value,
                "settings": field_meta.settings,
            }

            columns.append(column)

    # set the position of the columns
    for i in range(len(columns)):
        columns[i]["settings"]["position"] = i
    return columns


@patch(cls_method=True)
def instance_to_row(cls: ModelConverter, instance, model_class=None):
    """Convert a Pydantic model instance to a Ragas API row."""
    if model_class is None:
        model_class = instance.__class__

    row_cells = []
    model_data = instance.model_dump()

    for field_name, field_info in model_class.model_fields.items():
        if field_name in model_data:
            value = model_data[field_name]
            # Get the field's type annotation
            annotation = field_info.annotation

            # Special handling for MetricResult fields
            if isinstance(value, MetricResult):
                # Process the result value
                field_meta = infer_metric_result_type(value)
                processed_value = value._result

                # Add result cell
                row_cells.append({"column_id": field_name, "data": processed_value})

                # Add reason cell
                row_cells.append(
                    {"column_id": f"{field_name}_reason", "data": value.reason}
                )
            else:
                # Regular field handling
                field_meta = infer_field_type(annotation, field_info)

                # Special handling for various types
                if field_meta.type == ColumnType.MULTI_SELECT and isinstance(
                    value, list
                ):
                    # Convert list to string format accepted by API
                    processed_value = value
                elif field_meta.type == ColumnType.DATE and isinstance(
                    value, (datetime, date)
                ):
                    # Format date as string
                    processed_value = value.isoformat()
                else:
                    processed_value = value

                row_cells.append({"column_id": field_name, "data": processed_value})

    return {"data": row_cells}


@patch(cls_method=True)
def instances_to_rows(cls: ModelConverter, instances, model_class=None):
    """Convert multiple Pydantic model instances to Ragas API rows."""
    if not instances:
        return []

    if model_class is None and instances:
        model_class = instances[0].__class__

    return [cls.instance_to_row(instance, model_class) for instance in instances]
