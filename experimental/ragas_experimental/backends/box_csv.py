"""Box CSV backend implementation for projects and datasets."""

import csv
import io
import json
import logging
import os
import typing as t
import uuid
from typing import TYPE_CHECKING, Optional, get_origin

from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

from .utils import create_nano_id
from .base import DataTableBackend, ProjectBackend
from .config import BoxCSVConfig, BoxClientProtocol, BoxFolderProtocol, BoxFileProtocol

logger = logging.getLogger(__name__)

# Type-only imports for static analysis
if TYPE_CHECKING:
    from boxsdk import BoxAPIException, Client
    from boxsdk.object.folder import Folder
    from boxsdk.object.file import File
else:
    # Runtime imports with fallbacks
    try:
        from boxsdk import BoxAPIException, Client
        from boxsdk.object.folder import Folder
        from boxsdk.object.file import File
    except ImportError:
        logger.warning(
            "Box SDK not available. Install with: pip install 'ragas_experimental[box]' to use Box backend."
        )
        # Create placeholder types for runtime
        from typing import Any

        BoxAPIException = Any
        Client = Any
        Folder = Any
        File = Any


class BoxCSVDataTableBackend(DataTableBackend):
    """Box CSV implementation of DataTableBackend."""

    def __init__(
        self,
        box_client: BoxClientProtocol,
        project_folder_id: str,
        dataset_id: str,
        dataset_name: str,
        datatable_type: t.Literal["datasets", "experiments"],
    ):
        self.box_client = box_client
        self.project_folder_id = project_folder_id
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.datatable_type = datatable_type
        self.dataset = None
        self._csv_file: Optional[BoxFileProtocol] = None

    def _is_json_serializable_type(self, field_type):
        """Check if field needs JSON serialization."""
        origin = get_origin(field_type)
        return origin in (list, dict) or field_type in (list, dict)

    def initialize(self, dataset: t.Any) -> None:
        """Initialize the backend with dataset information."""
        self.dataset = dataset
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Create the CSV file on Box if it doesn't exist."""
        try:
            # Get or create the datatable folder (datasets/experiments)
            datatable_folder = self._get_or_create_folder(
                self.project_folder_id, self.datatable_type
            )

            # Check if CSV file already exists
            csv_filename = f"{self.dataset_name}.csv"
            self._csv_file = self._get_file_in_folder(datatable_folder, csv_filename)

            if self._csv_file is None:
                # Create CSV with headers
                if self.dataset is None:
                    raise ValueError(
                        "Dataset must be initialized before creating CSV headers"
                    )
                field_names = ["_row_id"] + list(
                    self.dataset.model.__annotations__.keys()
                )

                # Create CSV content with headers
                csv_content = io.StringIO()
                writer = csv.writer(csv_content)
                writer.writerow(field_names)
                csv_content.seek(0)

                # Upload to Box
                self._csv_file = datatable_folder.upload_stream(
                    csv_content, csv_filename
                )
                logger.info(f"Created CSV file on Box: {csv_filename}")

        except Exception as e:
            logger.error(f"Error ensuring CSV exists on Box: {e}")
            raise

    def _get_or_create_folder(self, parent_folder_id: str, folder_name: str) -> BoxFolderProtocol:
        """Get existing folder or create new one."""
        try:
            parent_folder = self.box_client.folder(parent_folder_id)

            # Check if folder already exists
            for item in parent_folder.get_items():
                if item.type == "folder" and item.name == folder_name:
                    return self.box_client.folder(item.id)

            # Create new folder
            new_folder = parent_folder.create_subfolder(folder_name)
            logger.info(f"Created folder on Box: {folder_name}")
            return new_folder

        except Exception as e:
            logger.error(f"Error creating/getting folder {folder_name}: {e}")
            raise

    def _get_file_in_folder(self, folder: BoxFolderProtocol, filename: str) -> Optional[BoxFileProtocol]:
        """Get file by name in folder, return None if not found."""
        try:
            for item in folder.get_items():
                if item.type == "file" and item.name == filename:
                    return self.box_client.file(item.id)
            return None
        except Exception as e:
            logger.error(f"Error searching for file {filename}: {e}")
            return None

    def get_column_mapping(self, model) -> t.Dict:
        """Get mapping between model fields and CSV columns."""
        return model.model_fields

    def load_entries(self, model_class):
        """Load all entries from the CSV file on Box."""
        if self._csv_file is None:
            return []

        try:
            # Download CSV content
            csv_content = self._csv_file.content().decode("utf-8")
            csv_reader = csv.DictReader(io.StringIO(csv_content))

            entries = []
            for row in csv_reader:
                try:
                    # Extract row_id and remove from model data
                    row_id = row.get("_row_id", str(uuid.uuid4()))

                    # Create a copy without _row_id for model instantiation
                    model_data = {k: v for k, v in row.items() if k != "_row_id"}

                    # Convert types as needed
                    typed_row = {}
                    for field, value in model_data.items():
                        if field in model_class.model_fields:
                            field_type = model_class.model_fields[field].annotation

                            try:
                                if not value:  # Handle empty strings
                                    typed_row[field] = None
                                elif self._is_json_serializable_type(field_type):
                                    # Deserialize JSON for lists/dicts
                                    typed_row[field] = json.loads(value)
                                elif field_type is int:
                                    typed_row[field] = int(value)
                                elif field_type is float:
                                    typed_row[field] = float(value)
                                elif field_type is bool:
                                    typed_row[field] = value.lower() in (
                                        "true",
                                        "t", 
                                        "yes",
                                        "y",
                                        "1",
                                    )
                                else:
                                    typed_row[field] = value
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.warning(f"Failed to convert field {field}='{value}' to {field_type}: {e}")
                                typed_row[field] = value  # Fallback to string

                    # Create model instance
                    entry = model_class(**typed_row)

                    # Set the row ID from CSV
                    entry._row_id = row_id

                    entries.append(entry)
                except Exception as e:
                    logger.error(f"Error loading row from CSV: {e}")

            return entries

        except Exception as e:
            logger.error(f"Error loading entries from Box CSV: {e}")
            return []

    def append_entry(self, entry) -> str:
        """Add a new entry to the CSV file on Box and return a generated ID."""
        try:
            # Load existing entries
            existing_entries = self.load_entries(entry.__class__)

            # Generate a row ID if needed
            row_id = getattr(entry, "_row_id", None) or str(uuid.uuid4())
            entry._row_id = row_id

            # Add new entry
            existing_entries.append(entry)

            # Write all entries back to Box
            self._write_entries_to_box(existing_entries)

            return row_id

        except Exception as e:
            logger.error(f"Error appending entry to Box CSV: {e}")
            raise

    def update_entry(self, entry) -> bool:
        """Update an existing entry in the CSV file on Box."""
        try:
            # Load existing entries
            existing_entries = self.load_entries(entry.__class__)

            # Find and update the entry
            updated = False
            for i, e in enumerate(existing_entries):
                if (
                    hasattr(e, "_row_id")
                    and hasattr(entry, "_row_id")
                    and e._row_id == entry._row_id
                ):
                    existing_entries[i] = entry
                    updated = True
                    break

            # If entry wasn't found, append it
            if not updated:
                existing_entries.append(entry)

            # Write all entries back to Box
            self._write_entries_to_box(existing_entries)

            return True

        except Exception as e:
            logger.error(f"Error updating entry in Box CSV: {e}")
            return False

    def delete_entry(self, entry_id) -> bool:
        """Delete an entry from the CSV file on Box."""
        try:
            if self.dataset is None:
                raise ValueError("Dataset must be initialized")

            # Filter out the entry to delete
            entries_to_keep = []
            for e in self.dataset._entries:
                if not (hasattr(e, "_row_id") and e._row_id == entry_id):
                    entries_to_keep.append(e)

            # Write remaining entries back to Box
            self._write_entries_to_box(entries_to_keep)

            return True

        except Exception as e:
            logger.error(f"Error deleting entry from Box CSV: {e}")
            return False

    def _write_entries_to_box(self, entries):
        """Write all entries to the CSV file on Box."""
        if self._csv_file is None:
            raise ValueError("CSV file not initialized")

        try:
            # Create CSV content
            csv_content = io.StringIO()

            if not entries:
                # If no entries, just create headers
                if self.dataset is None:
                    raise ValueError("Dataset must be initialized")
                field_names = ["_row_id"] + list(self.dataset.model.model_fields.keys())
                writer = csv.DictWriter(csv_content, fieldnames=field_names)
                writer.writeheader()
            else:
                # Get field names including _row_id
                field_names = ["_row_id"] + list(
                    entries[0].__class__.model_fields.keys()
                )
                writer = csv.DictWriter(csv_content, fieldnames=field_names)
                writer.writeheader()

                for entry in entries:
                    # Create a dict with model data + row_id, handling JSON serialization
                    entry_dict = {}
                    for field_name, field_value in entry.model_dump().items():
                        field_type = entry.__class__.model_fields[field_name].annotation
                        if self._is_json_serializable_type(field_type):
                            entry_dict[field_name] = json.dumps(field_value) if field_value is not None else ""
                        else:
                            entry_dict[field_name] = field_value
                    entry_dict["_row_id"] = getattr(entry, "_row_id", str(uuid.uuid4()))
                    writer.writerow(entry_dict)

            csv_content.seek(0)

            # Upload new version to Box
            self._csv_file.update_contents_with_stream(csv_content)
            logger.debug(f"Updated CSV file on Box with {len(entries)} entries")

        except Exception as e:
            logger.error(f"Error writing entries to Box CSV: {e}")
            raise

    def get_entry_by_field(
        self, field_name, field_value, model_class
    ) -> t.Optional[t.Any]:
        """Get an entry by field value."""
        entries = self.load_entries(model_class)

        for entry in entries:
            if hasattr(entry, field_name) and getattr(entry, field_name) == field_value:
                return entry

        return None


class BoxCSVProjectBackend(ProjectBackend):
    """Box CSV implementation of ProjectBackend."""

    def __init__(self, config: BoxCSVConfig):
        """Initialize Box backend with authenticated client.

        Args:
            config: BoxCSVConfig object containing authenticated Box client.
        """
        self.config = config
        self.box_client: BoxClientProtocol = config.client
        self.project_id: Optional[str] = None
        self.project_folder: Optional[BoxFolderProtocol] = None
    
    @classmethod
    def from_jwt_file(cls, config_file_path: str, 
                      root_folder_id: str = "0") -> 'BoxCSVProjectBackend':
        """Convenience constructor for JWT authentication from config file.
        
        Args:
            config_file_path: Path to Box JWT configuration file
            root_folder_id: Box folder ID to use as root (defaults to "0")
            
        Returns:
            BoxCSVProjectBackend instance with authenticated client
        """
        try:
            # Import here to avoid dependency issues if not available
            from boxsdk.auth.jwt_auth import JWTAuth
            from boxsdk import Client
        except ImportError:
            raise ImportError(
                "Box SDK not available. Install with: pip install 'ragas_experimental[box]'"
            )
        
        auth = JWTAuth.from_settings_file(config_file_path)
        client = Client(auth)
        config = BoxCSVConfig(client=client, root_folder_id=root_folder_id)
        return cls(config)
    
    @classmethod
    def from_developer_token(cls, token: str,
                            root_folder_id: str = "0") -> 'BoxCSVProjectBackend':
        """Convenience constructor for developer token (testing only).
        
        Args:
            token: Box developer token
            root_folder_id: Box folder ID to use as root (defaults to "0")
            
        Returns:
            BoxCSVProjectBackend instance with authenticated client
        """
        try:
            # Import here to avoid dependency issues if not available
            from boxsdk.auth.oauth2 import OAuth2
            from boxsdk import Client
        except ImportError:
            raise ImportError(
                "Box SDK not available. Install with: pip install 'ragas_experimental[box]'"
            )
        
        oauth = OAuth2(
            client_id='not_needed_for_dev_token',
            client_secret='not_needed_for_dev_token',
            access_token=token
        )
        client = Client(oauth)
        config = BoxCSVConfig(client=client, root_folder_id=root_folder_id)
        return cls(config)
    
    @classmethod  
    def from_oauth2(cls, client_id: str, client_secret: str, access_token: str,
                    refresh_token: Optional[str] = None, root_folder_id: str = "0") -> 'BoxCSVProjectBackend':
        """Convenience constructor for OAuth2 authentication.
        
        Args:
            client_id: Box application client ID
            client_secret: Box application client secret
            access_token: User access token
            refresh_token: Optional refresh token
            root_folder_id: Box folder ID to use as root (defaults to "0")
            
        Returns:
            BoxCSVProjectBackend instance with authenticated client
        """
        try:
            # Import here to avoid dependency issues if not available
            from boxsdk.auth.oauth2 import OAuth2
            from boxsdk import Client
        except ImportError:
            raise ImportError(
                "Box SDK not available. Install with: pip install 'ragas_experimental[box]'"
            )
        
        oauth = OAuth2(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            refresh_token=refresh_token
        )
        client = Client(oauth)
        config = BoxCSVConfig(client=client, root_folder_id=root_folder_id)
        return cls(config)


    def initialize(self, project_id: str, **kwargs):
        """Initialize the backend with project information."""
        self.project_id = project_id

        # Get or create project folder
        root_folder_id = self.config.root_folder_id
        root_folder = self.box_client.folder(root_folder_id)

        # Check if project folder exists
        project_folder = None
        for item in root_folder.get_items():
            if item.type == "folder" and item.name == project_id:
                project_folder = self.box_client.folder(item.id)
                break

        # Create project folder if it doesn't exist
        if project_folder is None:
            project_folder = root_folder.create_subfolder(project_id)
            logger.info(f"Created project folder on Box: {project_id}")

        self.project_folder = project_folder
        self._create_project_structure()

    def _create_project_structure(self):
        """Create the folder structure for the project on Box."""
        if self.project_folder is None:
            raise ValueError("Project folder not initialized")

        # Create datasets and experiments folders
        for folder_name in ["datasets", "experiments"]:
            folder_exists = False
            for item in self.project_folder.get_items():
                if item.type == "folder" and item.name == folder_name:
                    folder_exists = True
                    break

            if not folder_exists:
                self.project_folder.create_subfolder(folder_name)
                logger.info(f"Created {folder_name} folder on Box")

    def create_dataset(self, name: str, model: t.Type[BaseModel]) -> str:
        """Create a new dataset and return its ID."""
        dataset_id = create_nano_id()
        return dataset_id

    def create_experiment(self, name: str, model: t.Type[BaseModel]) -> str:
        """Create a new experiment and return its ID."""
        experiment_id = create_nano_id()
        return experiment_id

    def list_datasets(self) -> t.List[t.Dict]:
        """List all datasets in the project."""
        if self.project_folder is None:
            return []

        try:
            datasets = []

            # Find datasets folder
            datasets_folder = None
            for item in self.project_folder.get_items():
                if item.type == "folder" and item.name == "datasets":
                    datasets_folder = self.box_client.folder(item.id)
                    break

            if datasets_folder is None:
                return []

            # List CSV files in datasets folder
            for item in datasets_folder.get_items():
                if item.type == "file" and item.name.endswith(".csv"):
                    name = os.path.splitext(item.name)[0]
                    datasets.append(
                        {
                            "id": create_nano_id(),
                            "name": name,
                        }
                    )

            return datasets

        except Exception as e:
            logger.error(f"Error listing datasets from Box: {e}")
            return []

    def list_experiments(self) -> t.List[t.Dict]:
        """List all experiments in the project."""
        if self.project_folder is None:
            return []

        try:
            experiments = []

            # Find experiments folder
            experiments_folder = None
            for item in self.project_folder.get_items():
                if item.type == "folder" and item.name == "experiments":
                    experiments_folder = self.box_client.folder(item.id)
                    break

            if experiments_folder is None:
                return []

            # List CSV files in experiments folder
            for item in experiments_folder.get_items():
                if item.type == "file" and item.name.endswith(".csv"):
                    name = os.path.splitext(item.name)[0]
                    experiments.append(
                        {
                            "id": create_nano_id(),
                            "name": name,
                        }
                    )

            return experiments

        except Exception as e:
            logger.error(f"Error listing experiments from Box: {e}")
            return []

    def get_dataset_backend(
        self, dataset_id: str, name: str, model: t.Type[BaseModel]
    ) -> DataTableBackend:
        """Get a DataTableBackend instance for a specific dataset."""
        if self.project_folder is None:
            raise ValueError("Backend not properly initialized")

        return BoxCSVDataTableBackend(
            box_client=self.box_client,
            project_folder_id=self.project_folder.object_id,
            dataset_id=dataset_id,
            dataset_name=name,
            datatable_type="datasets",
        )

    def get_experiment_backend(
        self, experiment_id: str, name: str, model: t.Type[BaseModel]
    ) -> DataTableBackend:
        """Get a DataTableBackend instance for a specific experiment."""
        if self.project_folder is None:
            raise ValueError("Backend not properly initialized")

        return BoxCSVDataTableBackend(
            box_client=self.box_client,
            project_folder_id=self.project_folder.object_id,
            dataset_id=experiment_id,
            dataset_name=name,
            datatable_type="experiments",
        )

    def get_dataset_by_name(
        self, name: str, model: t.Type[BaseModel]
    ) -> t.Tuple[str, DataTableBackend]:
        """Get dataset ID and backend by name."""
        if self.project_folder is None:
            raise ValueError("Backend not initialized")

        try:
            # Check if dataset exists
            datasets_folder = None
            for item in self.project_folder.get_items():
                if item.type == "folder" and item.name == "datasets":
                    datasets_folder = self.box_client.folder(item.id)
                    break

            if datasets_folder is None:
                raise ValueError("Datasets folder not found")

            # Look for CSV file
            csv_exists = False
            for item in datasets_folder.get_items():
                if item.type == "file" and item.name == f"{name}.csv":
                    csv_exists = True
                    break

            if not csv_exists:
                raise ValueError(f"Dataset '{name}' does not exist")

            # Create dataset instance
            dataset_id = create_nano_id()
            backend = self.get_dataset_backend(dataset_id, name, model)

            return dataset_id, backend

        except Exception as e:
            logger.error(f"Error getting dataset by name: {e}")
            raise

    def get_experiment_by_name(
        self, name: str, model: t.Type[BaseModel]
    ) -> t.Tuple[str, DataTableBackend]:
        """Get experiment ID and backend by name."""
        if self.project_folder is None:
            raise ValueError("Backend not initialized")

        try:
            # Check if experiment exists
            experiments_folder = None
            for item in self.project_folder.get_items():
                if item.type == "folder" and item.name == "experiments":
                    experiments_folder = self.box_client.folder(item.id)
                    break

            if experiments_folder is None:
                raise ValueError("Experiments folder not found")

            # Look for CSV file
            csv_exists = False
            for item in experiments_folder.get_items():
                if item.type == "file" and item.name == f"{name}.csv":
                    csv_exists = True
                    break

            if not csv_exists:
                raise ValueError(f"Experiment '{name}' does not exist")

            # Create experiment instance
            experiment_id = create_nano_id()
            backend = self.get_experiment_backend(experiment_id, name, model)

            return experiment_id, backend

        except Exception as e:
            logger.error(f"Error getting experiment by name: {e}")
            raise
