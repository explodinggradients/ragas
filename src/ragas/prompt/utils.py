import copy
import typing as t

from pydantic import BaseModel


def get_all_strings(obj: t.Any) -> list[str]:
    """
    Get all strings in the objects.
    """
    strings = []

    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, BaseModel):
        for field_value in obj.model_dump().values():
            strings.extend(get_all_strings(field_value))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            strings.extend(get_all_strings(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            strings.extend(get_all_strings(value))

    return strings


def update_strings(obj: t.Any, old_strings: list[str], new_strings: list[str]) -> t.Any:
    """
    Replace strings in the object with new strings.
    Example Usage:
    ```
    old_strings = ["old1", "old2", "old3"]
    new_strings = ["new1", "new2", "new3"]
    obj = {"a": "old1", "b": "old2", "c": ["old1", "old2", "old3"], "d": {"e": "old2"}}
    update_strings(obj, old_strings, new_strings)
    ```
    """
    if len(old_strings) != len(new_strings):
        raise ValueError("The number of old and new strings must be the same")

    def replace_string(s: str) -> str:
        for old, new in zip(old_strings, new_strings):
            if s == old:
                return new
        return s

    if isinstance(obj, str):
        return replace_string(obj)
    elif isinstance(obj, BaseModel):
        new_obj = copy.deepcopy(obj)
        for field in new_obj.model_fields:
            setattr(
                new_obj,
                field,
                update_strings(getattr(new_obj, field), old_strings, new_strings),
            )
        return new_obj
    elif isinstance(obj, list):
        return [update_strings(item, old_strings, new_strings) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(update_strings(item, old_strings, new_strings) for item in obj)
    elif isinstance(obj, dict):
        return {k: update_strings(v, old_strings, new_strings) for k, v in obj.items()}

    return copy.deepcopy(obj)
