"""Local JSONL backend implementation for projects and datasets."""

import json
import re
import typing as t
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

from .base import BaseBackend

_TYPE_TAG_KEY = "__ragas_type__"
_TYPE_TAG_DATETIME = "datetime"
_TYPE_TAG_DATE = "date"

# Every row written by this version of the backend carries a top-level
# ``__ragas_format__`` field. The load path uses it to distinguish
# "authoritative new-format" rows from legacy rows saved by older versions:
# only legacy rows fall back to the regex-anchored ISO 8601 coercion.
_FORMAT_KEY = "__ragas_format__"
_FORMAT_VERSION = "v2"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$"
)


class LocalJSONLBackend(BaseBackend):
    """File-based backend using JSONL format for local storage.

    Stores datasets and experiments as JSONL files (one JSON object per line).
    Preserves data types and supports complex nested structures including
    datetime objects, lists, and nested dictionaries.

    On-disk format:
        Records are written as JSONL (one JSON object per line). ``date`` and
        ``datetime`` values are tagged with a discriminator so the load path
        can restore them without relying on string shape heuristics:

            {"d": {"__ragas_type__": "date", "value": "2026-07-20"}}
            {"d": {"__ragas_type__": "datetime", "value": "2026-07-20T12:00:00"}}

        Strings, numbers, booleans, ``None``, lists, and nested dicts are
        written verbatim. Values that were not a ``date`` / ``datetime`` at
        save time stay as their original type on load — even if the string
        happens to look like an ISO 8601 date.

    Legacy compatibility:
        Records written by older versions of this backend do not carry the
        discriminator. For such records the load path uses a
        regex-anchored ISO 8601 fallback so that genuine ISO strings (e.g.
        ``"2026-07-20"``) are still coerced, but arbitrary user strings
        such as ``"hi-jk-lm-no"`` are left untouched.

    Directory Structure:
        root_dir/
        ├── datasets/
        │   ├── dataset1.jsonl
        │   └── dataset2.jsonl
        └── experiments/
            ├── experiment1.jsonl
            └── experiment2.jsonl

    Args:
        root_dir: Directory path for storing JSONL files

    Features:
        - Preserves Python data types (int, float, bool, None, str)
        - Round-trips ``date`` / ``datetime`` values via a tagged representation
        - Supports nested dictionaries and lists
        - Handles malformed JSON lines gracefully (skips with warning)
        - UTF-8 encoding for international text
        - Compact JSON formatting (no extra whitespace)

    Best For:
        - Complex data structures with nesting
        - Mixed data types and datetime objects
        - When data type preservation is important
        - Large datasets (streaming line-by-line processing)
    """

    def __init__(
        self,
        root_dir: str,
    ):
        self.root_dir = Path(root_dir)

    def _get_data_dir(self, data_type: str) -> Path:
        """Get the directory path for datasets or experiments."""
        return self.root_dir / data_type

    def _get_file_path(self, data_type: str, name: str) -> Path:
        """Get the full file path for a dataset or experiment."""
        return self._get_data_dir(data_type) / f"{name}.jsonl"

    @staticmethod
    def _serialize_with_type_tag(obj: t.Any) -> t.Any:
        """Tag ``datetime`` / ``date`` values with a discriminator on save.

        Other values pass through unchanged.
        """
        if isinstance(obj, datetime):
            return {_TYPE_TAG_KEY: _TYPE_TAG_DATETIME, "value": obj.isoformat()}
        if isinstance(obj, date):
            return {_TYPE_TAG_KEY: _TYPE_TAG_DATE, "value": obj.isoformat()}
        if isinstance(obj, dict):
            return {
                k: LocalJSONLBackend._serialize_with_type_tag(v) for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [LocalJSONLBackend._serialize_with_type_tag(item) for item in obj]
        return obj

    @staticmethod
    def _deserialize_with_type_tag(obj: t.Any) -> t.Any:
        """Restore tagged ``date`` / ``datetime`` values on load.

        Tagged values are restored to their original type. Anything else is
        returned untouched, including strings that happen to look like ISO
        8601 dates — the discriminator is the only signal we trust.
        """
        if isinstance(obj, dict):
            if _TYPE_TAG_KEY in obj and set(obj.keys()).issubset(
                {_TYPE_TAG_KEY, "value"}
            ):
                tag = obj.get(_TYPE_TAG_KEY)
                raw = obj.get("value")
                if not isinstance(raw, str):
                    return obj
                try:
                    if tag == _TYPE_TAG_DATETIME:
                        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    if tag == _TYPE_TAG_DATE:
                        return date.fromisoformat(raw)
                except (ValueError, TypeError):
                    return obj
                return obj
            return {
                k: LocalJSONLBackend._deserialize_with_type_tag(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [LocalJSONLBackend._deserialize_with_type_tag(item) for item in obj]
        return obj

    @classmethod
    def _deserialize_legacy_value(cls, obj: t.Any) -> t.Any:
        """Coerce strings that match ISO 8601 *exactly* on legacy records.

        Used only for records written by older versions of this backend that
        lacked the discriminator. Non-matching strings pass through.
        """
        if isinstance(obj, str):
            if _DATE_RE.match(obj):
                try:
                    return date.fromisoformat(obj)
                except ValueError:
                    return obj
            if _DATETIME_RE.match(obj):
                try:
                    return datetime.fromisoformat(obj.replace("Z", "+00:00"))
                except ValueError:
                    return obj
            return obj
        if isinstance(obj, dict):
            return {k: cls._deserialize_legacy_value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._deserialize_legacy_value(item) for item in obj]
        return obj

    def _load(self, data_type: str, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load data from JSONL file, raising FileNotFoundError if file doesn't exist."""
        file_path = self._get_file_path(data_type, name)

        if not file_path.exists():
            raise FileNotFoundError(
                f"No {data_type[:-1]} named '{name}' found at {file_path}"
            )

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    json_obj = json.loads(line)
                    if LocalJSONLBackend._is_new_format_row(json_obj):
                        # Authoritative new-format row: restore tagged values,
                        # leave everything else untouched, and strip the
                        # internal format marker.
                        json_obj = LocalJSONLBackend._deserialize_with_type_tag(
                            json_obj
                        )
                        json_obj.pop(_FORMAT_KEY, None)
                    else:
                        # Legacy row: only coerce values that *exactly* match
                        # an ISO 8601 regex; everything else is left as-is.
                        json_obj = LocalJSONLBackend._deserialize_legacy_value(json_obj)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    # Handle malformed JSON gracefully
                    print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                    continue

        return data

    @classmethod
    def _is_new_format_row(cls, obj: t.Any) -> bool:
        """Return True if obj is a top-level dict emitted by this backend."""
        return isinstance(obj, dict) and obj.get(_FORMAT_KEY) == _FORMAT_VERSION

    def _save(
        self,
        data_type: str,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]],
    ) -> None:
        """Save data to JSONL file, creating directory if needed."""
        file_path = self._get_file_path(data_type, name)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle empty data
        if not data:
            # Create empty JSONL file
            with open(file_path, "w", encoding="utf-8") as f:
                pass
            return

        # Write data to JSONL
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                # Tag date / datetime values so the load path can restore them
                # without string-shape heuristics, and stamp a row-level format
                # marker so the loader knows the file came from this version.
                serialized_item = self._serialize_with_type_tag(item)
                if isinstance(serialized_item, dict):
                    serialized_item[_FORMAT_KEY] = _FORMAT_VERSION
                # Write as JSON line
                json_line = json.dumps(
                    serialized_item, ensure_ascii=False, separators=(",", ":")
                )
                f.write(json_line + "\n")

    def _list(self, data_type: str) -> t.List[str]:
        """List all available datasets or experiments."""
        data_dir = self._get_data_dir(data_type)

        if not data_dir.exists():
            return []

        # Get all .jsonl files and return names without extension
        jsonl_files = [f.stem for f in data_dir.glob("*.jsonl")]
        return sorted(jsonl_files)

    # Public interface methods (required by BaseBackend)
    def load_dataset(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load dataset from JSONL file."""
        return self._load("datasets", name)

    def load_experiment(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load experiment from JSONL file."""
        return self._load("experiments", name)

    def save_dataset(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save dataset to JSONL file."""
        self._save("datasets", name, data, data_model)

    def save_experiment(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save experiment to JSONL file."""
        self._save("experiments", name, data, data_model)

    def list_datasets(self) -> t.List[str]:
        """List all dataset names."""
        return self._list("datasets")

    def list_experiments(self) -> t.List[str]:
        """List all experiment names."""
        return self._list("experiments")

    def __repr__(self) -> str:
        return f"LocalJSONLBackend(root_dir='{self.root_dir}')"

    __str__ = __repr__
