__all__ = []

# Get version from setuptools_scm-generated file
try:
    from ._version import version as __version__
except ImportError:
    # Fallback for installed package
    from importlib.metadata import version as pkg_version, PackageNotFoundError

    try:
        __version__ = pkg_version("ragas_experimental")
    except PackageNotFoundError:
        __version__ = "unknown"

from .project.core import Project
import ragas_experimental.model.notion_typing as nmt
from .model.notion_model import NotionModel
from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

# just import to run the module
import ragas_experimental.project.datasets
import ragas_experimental.project.experiments
import ragas_experimental.project.comparison

__all__ = ["Project", "NotionModel", "nmt", "BaseModel"]
