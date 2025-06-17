---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

# Project

> Use this class to represent the AI project that we are working on and to interact with datasets and experiments in it.

```python
# | default_exp project.core
```

```python
# | hide
from nbdev.showdoc import *
```

```python

from ragas_experimental.model.notion_model import NotionModel
```

```python
# | export
import typing as t
import os
import asyncio

from fastcore.utils import patch
from pydantic import BaseModel

from ragas_experimental.backends.factory import RagasApiClientFactory
from ragas_experimental.backends.ragas_api_client import RagasApiClient
import ragas_experimental.typing as rt
from ragas_experimental.utils import async_to_sync, create_nano_id
from ragas_experimental.dataset import Dataset
from ragas_experimental.experiment import Experiment
```

```python
# | export
class Project:
    def __init__(
        self,
        project_id: str,
        backend: t.Literal["ragas_api", "local"] = "local",
        root_dir: t.Optional[str] = None,
        ragas_api_client: t.Optional[RagasApiClient] = None,
    ):
        self.project_id = project_id
        if backend == "local":
            self._root_dir = root_dir
        elif backend == "ragas_api":
            if ragas_api_client is None:
                self._ragas_api_client = RagasApiClientFactory.create()
            else:
                self._ragas_api_client = ragas_api_client
        else:
            raise ValueError(f"Invalid backend: {backend}")
        # create the project
        if backend == "ragas_api":
            try:
                sync_version = async_to_sync(self._ragas_api_client.get_project)
                existing_project = sync_version(project_id=self.project_id)
                self.project_id = existing_project["id"]
                self.name = existing_project["title"]
                self.description = existing_project["description"]
            except Exception as e:
                raise e
        elif backend == "local":
            self.name = self.project_id
            # create a new folder in the root_dir/project_id
            self._root_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        ragas_api_client: t.Optional[RagasApiClient] = None,
    ):
        ragas_api_client = RagasApiClientFactory.create()
        sync_version = async_to_sync(ragas_api_client.create_project)
        new_project = sync_version(title=name, description=description)
        return cls(new_project["id"], ragas_api_client)

    def delete(self):
        sync_version = async_to_sync(self._ragas_api_client.delete_project)
        sync_version(project_id=self.project_id)
        print("Project deleted!")

    def __repr__(self):
        return f"Project(name='{self.name}')"
```

```python
RAGAS_APP_TOKEN = "api-key"
RAGAS_API_BASE_URL = "https://api.dev.app.ragas.io"

os.environ["RAGAS_APP_TOKEN"] = RAGAS_APP_TOKEN
os.environ["RAGAS_API_BASE_URL"] = RAGAS_API_BASE_URL
```

```python
#project = Project.create("Demo Project")
project = Project(project_id="1ef0843b-231f-4a2c-b64d-d39bcee9d830")
project
```

```python
# | export
@patch(cls_method=True)
def get(cls: Project, name: str, ragas_api_client: t.Optional[RagasApiClient] = None) -> Project:
    """Get an existing project by name."""
    # Search for project with given name
    if ragas_api_client is None:
        ragas_api_client = RagasApiClientFactory.create()

    # get the project by name
    sync_version = async_to_sync(ragas_api_client.get_project_by_name)
    project_info = sync_version(
        project_name=name
    )

    # Return Project instance
    return Project(
        project_id=project_info["id"],
        ragas_api_client=ragas_api_client,
    )
```

```python
Project.get("SuperMe")
```

```python
#project.delete()
```
