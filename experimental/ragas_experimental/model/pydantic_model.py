"""An Extended version of Pydantics `BaseModel` for some ragas specific stuff"""

__all__ = ["ExtendedPydanticBaseModel"]

import typing as t

from pydantic import BaseModel, PrivateAttr

from ..typing import FieldMeta as RagasFieldMeta


class ExtendedPydanticBaseModel(BaseModel):
    """Extended Pydantic BaseModel with database integration capabilities"""

    # Private attribute for storing the database row_id
    _row_id: t.Optional[int] = PrivateAttr(default=None)

    # Class variable for storing column mapping overrides
    __column_mapping__: t.ClassVar[t.Dict[str, str]] = {}

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize column mapping if not already defined
        if not self.__class__.__column_mapping__:
            self._initialize_column_mapping()

    @classmethod
    def _initialize_column_mapping(cls):
        """Initialize mapping from field names to column IDs."""
        for field_name, field_info in cls.model_fields.items():
            # Check if field has Column metadata (for Pydantic v2)
            column_id = None
            for extra in field_info.metadata or []:
                if isinstance(extra, RagasFieldMeta) and extra.id:
                    column_id = extra.id
                    break

            # If no Column metadata found, use field name as column ID
            if not column_id:
                column_id = field_name

            cls.__column_mapping__[field_name] = column_id

            # check if the field is a MetricResult
            if cls._is_metric_result_field(field_info.annotation):
                # add additional mapping for the metric result
                reason_field_name = f"{field_name}_reason"
                reason_column_id = f"{column_id}_reason"
                cls.__column_mapping__[reason_field_name] = reason_column_id

    @staticmethod
    def _is_metric_result_field(annotation):
        """Check if a field annotation represents a MetricResult."""
        # Direct import of MetricResult
        from ragas_experimental.metric.result import MetricResult

        # Check if annotation is or references MetricResult
        return (
            annotation is MetricResult
            or (
                hasattr(annotation, "__origin__")
                and annotation.__origin__ is MetricResult
            )
            or (
                hasattr(annotation, "__class__")
                and annotation.__class__ is MetricResult
            )
        )

    @classmethod
    def get_column_id(cls, field_name: str) -> str:
        """Get the column ID for a given field name."""
        if field_name not in cls.__column_mapping__:
            raise ValueError(f"No column mapping found for field {field_name}")
        return cls.__column_mapping__[field_name]

    @classmethod
    def set_column_id(cls, field_name: str, column_id: str):
        """Set the column ID for a given field name."""
        if field_name not in cls.model_fields:
            raise ValueError(f"Field {field_name} not found in model")
        cls.__column_mapping__[field_name] = column_id

    def get_db_field_mapping(self) -> t.Dict[str, str]:
        """Get a mapping from field names to column IDs for this model."""
        return self.__class__.__column_mapping__
