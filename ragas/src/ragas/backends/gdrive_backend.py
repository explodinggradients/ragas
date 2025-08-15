"""Google Drive backend for storing datasets and experiments in Google Sheets."""

import json
import logging
import os
import typing as t

from pydantic import BaseModel

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials as UserCredentials
    from google.oauth2.service_account import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False

    # Define stub classes for type checking when imports fail
    Request = type("Request", (), {})
    UserCredentials = type("UserCredentials", (), {})
    Credentials = type("Credentials", (), {})
    InstalledAppFlow = type("InstalledAppFlow", (), {})
    HttpError = type("HttpError", (Exception,), {})

    def build(*args, **kwargs):
        raise ImportError("Google API client not available")


from .base import BaseBackend

logger = logging.getLogger(__name__)


class GDriveBackend(BaseBackend):
    """Backend for storing datasets and experiments in Google Drive using Google Sheets.

    This backend stores datasets and experiments as Google Sheets within a specified
    Google Drive folder. Each dataset/experiment becomes a separate spreadsheet.

    Directory Structure in Google Drive:
        root_folder/
        ├── datasets/
        │   ├── dataset1.gsheet
        │   └── dataset2.gsheet
        └── experiments/
            ├── experiment1.gsheet
            └── experiment2.gsheet

    Args:
        folder_id: The ID of the Google Drive folder to store data
        credentials_path: Path to OAuth credentials JSON file (optional)
        service_account_path: Path to service account JSON file (optional)
        token_path: Path to store OAuth token (default: "token.json")

    Authentication:
        Supports both OAuth and service account authentication.
        - OAuth: Requires user interaction for first-time setup
        - Service Account: Automated authentication, requires folder sharing

    Environment Variables:
        - GDRIVE_CREDENTIALS_PATH: Path to OAuth credentials
        - GDRIVE_SERVICE_ACCOUNT_PATH: Path to service account JSON
        - GDRIVE_TOKEN_PATH: Path to OAuth token file
    """

    # Scopes needed for Google Drive and Sheets API
    SCOPES = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    def __init__(
        self,
        folder_id: str,
        credentials_path: t.Optional[str] = None,
        service_account_path: t.Optional[str] = None,
        token_path: t.Optional[str] = None,
    ):
        """Initialize the Google Drive backend.

        Args:
            folder_id: The ID of the Google Drive folder to store datasets/experiments
            credentials_path: Path to OAuth credentials JSON file
            service_account_path: Path to service account JSON file
            token_path: Path to store OAuth token
        """
        if not GDRIVE_AVAILABLE:
            raise ImportError(
                "Google Drive backend requires additional dependencies. "
                "Install with: pip install google-api-python-client google-auth google-auth-oauthlib"
            )

        self.folder_id = folder_id

        # Authentication paths
        self.credentials_path = credentials_path or os.getenv("GDRIVE_CREDENTIALS_PATH")
        self.service_account_path = service_account_path or os.getenv(
            "GDRIVE_SERVICE_ACCOUNT_PATH"
        )
        self.token_path = token_path or os.getenv("GDRIVE_TOKEN_PATH", "token.json")

        # Initialize Google API clients
        self._setup_auth()

        # Ensure folder structure exists
        self._ensure_folder_structure()

    def _setup_auth(self):
        """Set up authentication for Google APIs."""
        creds = None

        # Try service account authentication first
        if self.service_account_path and os.path.exists(self.service_account_path):
            creds = Credentials.from_service_account_file(  # type: ignore
                self.service_account_path, scopes=self.SCOPES
            )
            logger.debug("Using service account authentication")
        # Try OAuth authentication
        elif self.credentials_path and os.path.exists(self.credentials_path):
            # Load existing token if available
            if os.path.exists(self.token_path):
                creds = UserCredentials.from_authorized_user_file(  # type: ignore
                    self.token_path, self.SCOPES
                )

            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(  # type: ignore
                        self.credentials_path, self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save the credentials for the next run
                with open(self.token_path, "w") as token:
                    token.write(creds.to_json())
            logger.debug("Using OAuth authentication")
        else:
            raise ValueError(
                "No valid authentication method found. Please provide either:\n"
                "1. Service account JSON file path via service_account_path or GDRIVE_SERVICE_ACCOUNT_PATH\n"
                "2. OAuth credentials JSON file path via credentials_path or GDRIVE_CREDENTIALS_PATH"
            )

        # Build the services
        self.drive_service = build("drive", "v3", credentials=creds)
        self.sheets_service = build("sheets", "v4", credentials=creds)

    def _ensure_folder_structure(self):
        """Create the folder structure in Google Drive if it doesn't exist."""
        try:
            # Check if main folder exists
            folder_metadata = (
                self.drive_service.files().get(fileId=self.folder_id).execute()
            )
            logger.debug(f"Found main folder: {folder_metadata.get('name')}")
        except HttpError as e:
            if e.resp.status == 404:  # type: ignore
                raise ValueError(
                    f"Folder with ID {self.folder_id} not found or not accessible"
                )
            else:
                raise ValueError(
                    f"Failed to access folder with ID {self.folder_id}: {e}"
                )

        # Create datasets and experiments folders if they don't exist
        self.datasets_folder_id = self._get_or_create_folder("datasets", self.folder_id)
        self.experiments_folder_id = self._get_or_create_folder(
            "experiments", self.folder_id
        )

    def _get_or_create_folder(self, folder_name: str, parent_id: str) -> str:
        """Get existing folder ID or create new folder."""
        # Search for existing folder
        query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.drive_service.files().list(q=query).execute()
        folders = results.get("files", [])

        if folders:
            logger.debug(f"Found existing folder: {folder_name}")
            return folders[0]["id"]

        # Create new folder
        folder_metadata = {
            "name": folder_name,
            "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = self.drive_service.files().create(body=folder_metadata).execute()
        logger.debug(f"Created new folder: {folder_name}")
        return folder["id"]

    def _get_folder_id_for_type(self, data_type: str) -> str:
        """Get the folder ID for datasets or experiments."""
        if data_type == "datasets":
            return self.datasets_folder_id
        elif data_type == "experiments":
            return self.experiments_folder_id
        else:
            raise ValueError(
                f"Invalid data type: {data_type}. Must be 'datasets' or 'experiments'"
            )

    def _get_or_create_spreadsheet(self, name: str, data_type: str) -> str:
        """Get existing spreadsheet ID or create new spreadsheet."""
        folder_id = self._get_folder_id_for_type(data_type)
        spreadsheet_name = f"{name}.gsheet"

        # Search for existing spreadsheet
        query = f"name='{spreadsheet_name}' and '{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
        results = self.drive_service.files().list(q=query).execute()
        sheets = results.get("files", [])

        if sheets:
            logger.debug(f"Found existing spreadsheet: {spreadsheet_name}")
            return sheets[0]["id"]

        # Create new spreadsheet
        spreadsheet_metadata = {
            "name": spreadsheet_name,
            "parents": [folder_id],
            "mimeType": "application/vnd.google-apps.spreadsheet",
        }
        spreadsheet = (
            self.drive_service.files().create(body=spreadsheet_metadata).execute()
        )
        logger.debug(f"Created new spreadsheet: {spreadsheet_name}")
        return spreadsheet["id"]

    def _spreadsheet_exists(self, name: str, data_type: str) -> bool:
        """Check if a spreadsheet exists."""
        folder_id = self._get_folder_id_for_type(data_type)
        spreadsheet_name = f"{name}.gsheet"

        query = f"name='{spreadsheet_name}' and '{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
        results = self.drive_service.files().list(q=query).execute()
        return len(results.get("files", [])) > 0

    def _load_data_from_spreadsheet(
        self, name: str, data_type: str
    ) -> t.List[t.Dict[str, t.Any]]:
        """Load data from a Google Sheet."""
        if not self._spreadsheet_exists(name, data_type):
            # Use singular form for error message
            singular_type = (
                data_type.rstrip("s") if data_type.endswith("s") else data_type
            )
            raise FileNotFoundError(f"{singular_type.capitalize()} '{name}' not found")

        spreadsheet_id = self._get_or_create_spreadsheet(name, data_type)

        try:
            # Get all data from the sheet
            result = (
                self.sheets_service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range="A:Z")
                .execute()
            )

            values = result.get("values", [])
            if not values:
                return []

            # First row contains headers
            headers: t.List[str] = values[0]
            data_rows: t.List[t.List[str]] = values[1:]

            # Convert to list of dictionaries
            data: t.List[t.Dict[str, t.Any]] = []
            for row in t.cast(t.List[t.List[str]], data_rows):
                # Pad row with empty strings if shorter than headers
                padded_row = row + [""] * (len(headers) - len(row))

                # Skip empty rows
                if all(cell.strip() == "" for cell in padded_row):
                    continue

                row_dict: t.Dict[str, t.Any] = dict(zip(headers, padded_row))

                # Try to convert numeric strings back to numbers
                for key, value in row_dict.items():
                    if isinstance(value, str) and value.strip():
                        # Try int first, then float
                        try:
                            if "." not in value:
                                row_dict[key] = int(value)
                            else:
                                row_dict[key] = float(value)
                        except ValueError:
                            # Keep as string if conversion fails
                            pass

                data.append(row_dict)

            return data

        except HttpError as e:
            logger.error(
                f"Error loading data from spreadsheet {name}: HTTP {e.resp.status} - {e}"  # type: ignore
            )
            raise
        except Exception as e:
            logger.error(f"Error processing data from spreadsheet {name}: {e}")
            raise

    def _save_data_to_spreadsheet(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_type: str,
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save data to a Google Sheet."""
        spreadsheet_id = self._get_or_create_spreadsheet(name, data_type)

        if not data:
            # Clear the spreadsheet for empty data
            self.sheets_service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id, range="A:Z"
            ).execute()
            logger.debug(f"Cleared spreadsheet for empty {data_type} '{name}'")
            return

        # Get all unique keys from all dictionaries to create headers
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        headers = sorted(list(all_keys))

        # Prepare data for the sheet
        sheet_data = [headers]  # First row is headers

        for item in data:
            row = []
            for header in headers:
                value = item.get(header, "")
                # Convert to string for Google Sheets
                if isinstance(value, (list, dict)):
                    row.append(json.dumps(value))
                else:
                    row.append(str(value))
            sheet_data.append(row)

        try:
            # Clear existing data
            self.sheets_service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id, range="A:Z"
            ).execute()

            # Write new data
            self.sheets_service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range="A1",
                valueInputOption="RAW",
                body={"values": sheet_data},
            ).execute()

            logger.debug(f"Saved {len(data)} rows to {data_type} '{name}'")

        except HttpError as e:
            logger.error(
                f"Error saving data to spreadsheet {name}: HTTP {e.resp.status} - {e}"  # type: ignore
            )
            raise
        except Exception as e:
            logger.error(f"Error processing data for spreadsheet {name}: {e}")
            raise

    def _list_data_names(self, data_type: str) -> t.List[str]:
        """List all available dataset or experiment names."""
        folder_id = self._get_folder_id_for_type(data_type)

        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
        results = self.drive_service.files().list(q=query).execute()
        files: t.List[t.Dict[str, t.Any]] = results.get("files", [])

        # Extract names (remove .gsheet extension)
        names: t.List[str] = []
        for file in t.cast(t.List[t.Dict[str, t.Any]], files):
            name = file["name"]
            if name.endswith(".gsheet"):
                names.append(name[:-7])  # Remove .gsheet
            else:
                names.append(name)

        return sorted(names)

    # BaseBackend interface implementation

    def load_dataset(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load dataset by name."""
        return self._load_data_from_spreadsheet(name, "datasets")

    def load_experiment(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load experiment by name."""
        return self._load_data_from_spreadsheet(name, "experiments")

    def save_dataset(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save dataset with given name."""
        self._save_data_to_spreadsheet(name, data, "datasets", data_model)

    def save_experiment(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save experiment with given name."""
        self._save_data_to_spreadsheet(name, data, "experiments", data_model)

    def list_datasets(self) -> t.List[str]:
        """List all available dataset names."""
        return self._list_data_names("datasets")

    def list_experiments(self) -> t.List[str]:
        """List all available experiment names."""
        return self._list_data_names("experiments")

    def __repr__(self) -> str:
        return f"GDriveBackend(folder_id='{self.folder_id}')"

    __str__ = __repr__
