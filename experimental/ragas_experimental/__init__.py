# Get version from setuptools_scm-generated file
try:
    from ._version import version as __version__
except ImportError:
    # Fallback for installed package
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        __version__ = pkg_version("ragas_experimental")
    except PackageNotFoundError:
        __version__ = "unknown"

import ragas_experimental.model.notion_typing as nmt
from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

from .model.notion_model import NotionModel
from .project.core import Project

# Import the main Project class - decorators are added automatically in core.py

__all__ = ["Project", "NotionModel", "nmt", "BaseModel"]
