"""Google Drive backend for storing datasets in Google Sheets."""

import typing as t
import os
import json
import uuid
from datetime import datetime

try:
    from googleapiclient.discovery import build
    from google.oauth2.service_account import Credentials
    from google.oauth2.credentials import Credentials as UserCredentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    raise ImportError(
        "Google Drive backend requires google-api-python-client and google-auth. "
        "Install with: pip install google-api-python-client google-auth google-auth-oauthlib"
    )

from .base import DatasetBackend
from ..utils import create_nano_id


class GDriveBackend(DatasetBackend):
    """Backend for storing datasets in Google Drive using Google Sheets."""
    
    # Scopes needed for Google Drive and Sheets API
    SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/spreadsheets'
    ]
    
    def __init__(
        self,
        folder_id: str,
        project_id: str,
        dataset_id: str,
        dataset_name: str,
        type: t.Literal["datasets", "experiments"],
        credentials_path: t.Optional[str] = None,
        service_account_path: t.Optional[str] = None,
        token_path: t.Optional[str] = None,
    ):
        """Initialize the Google Drive backend.
        
        Args:
            folder_id: The ID of the Google Drive folder to store datasets
            project_id: The ID of the project
            dataset_id: The ID of the dataset
            dataset_name: The name of the dataset
            type: Type of data (datasets or experiments)
            credentials_path: Path to OAuth credentials JSON file
            service_account_path: Path to service account JSON file
            token_path: Path to store OAuth token
        """
        self.folder_id = folder_id
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.type = type
        self.dataset = None
        
        # Authentication paths
        self.credentials_path = credentials_path or os.getenv("GDRIVE_CREDENTIALS_PATH")
        self.service_account_path = service_account_path or os.getenv("GDRIVE_SERVICE_ACCOUNT_PATH")
        self.token_path = token_path or os.getenv("GDRIVE_TOKEN_PATH", "token.json")
        
        # Initialize Google API clients
        self._setup_auth()
        
    def __str__(self):
        return f"GDriveBackend(folder_id={self.folder_id}, project_id={self.project_id}, dataset_id={self.dataset_id}, dataset_name={self.dataset_name})"

    def __repr__(self):
        return self.__str__()
    
    def _setup_auth(self):
        """Set up authentication for Google APIs."""
        creds = None
        
        # Try service account authentication first
        if self.service_account_path and os.path.exists(self.service_account_path):
            creds = Credentials.from_service_account_file(
                self.service_account_path, scopes=self.SCOPES
            )
        # Try OAuth authentication
        elif self.credentials_path and os.path.exists(self.credentials_path):
            # Load existing token if available
            if os.path.exists(self.token_path):
                creds = UserCredentials.from_authorized_user_file(self.token_path, self.SCOPES)
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())
        else:
            raise ValueError(
                "No valid authentication method found. Please provide either:\n"
                "1. Service account JSON file path via service_account_path or GDRIVE_SERVICE_ACCOUNT_PATH\n"
                "2. OAuth credentials JSON file path via credentials_path or GDRIVE_CREDENTIALS_PATH"
            )
        
        # Build the services
        self.drive_service = build('drive', 'v3', credentials=creds)
        self.sheets_service = build('sheets', 'v4', credentials=creds)

    def initialize(self, dataset):
        """Initialize the backend with the dataset instance."""
        self.dataset = dataset
        
        # Ensure the project folder structure exists
        self._ensure_folder_structure()
        
        # Ensure the spreadsheet exists
        self._ensure_spreadsheet_exists()

    def _ensure_folder_structure(self):
        """Create the folder structure in Google Drive if it doesn't exist."""
        try:
            # Check if main folder exists
            folder_metadata = self.drive_service.files().get(fileId=self.folder_id).execute()
        except:
            raise ValueError(f"Folder with ID {self.folder_id} not found or not accessible")
        
        # Create project folder if it doesn't exist
        project_folder_id = self._get_or_create_folder(self.project_id, self.folder_id)
        
        # Create type folder (datasets/experiments) if it doesn't exist
        self.type_folder_id = self._get_or_create_folder(self.type, project_folder_id)

    def _get_or_create_folder(self, folder_name: str, parent_id: str) -> str:
        """Get existing folder ID or create new folder."""
        # Search for existing folder
        query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.drive_service.files().list(q=query).execute()
        folders = results.get('files', [])
        
        if folders:
            return folders[0]['id']
        
        # Create new folder
        folder_metadata = {
            'name': folder_name,
            'parents': [parent_id],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = self.drive_service.files().create(body=folder_metadata).execute()
        return folder['id']

    def _ensure_spreadsheet_exists(self):
        """Create the Google Sheet if it doesn't exist."""
        spreadsheet_name = f"{self.dataset_name}.gsheet"
        
        # Search for existing spreadsheet
        query = f"name='{spreadsheet_name}' and '{self.type_folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
        results = self.drive_service.files().list(q=query).execute()
        sheets = results.get('files', [])
        
        if sheets:
            self.spreadsheet_id = sheets[0]['id']
            # Always ensure headers are correct for existing sheets
            self._initialize_spreadsheet_headers()
        else:
            # Create new spreadsheet
            spreadsheet_metadata = {
                'name': spreadsheet_name,
                'parents': [self.type_folder_id],
                'mimeType': 'application/vnd.google-apps.spreadsheet'
            }
            spreadsheet = self.drive_service.files().create(body=spreadsheet_metadata).execute()
            self.spreadsheet_id = spreadsheet['id']
            
            # Initialize with headers
            self._initialize_spreadsheet_headers()

    def _initialize_spreadsheet_headers(self):
        """Initialize the spreadsheet with headers."""
        if self.dataset is not None:
            try:
                # Include _row_id in the headers
                expected_headers = ["_row_id"] + list(self.dataset.model.model_fields.keys())
                
                # Check if headers are already correct
                try:
                    result = self.sheets_service.spreadsheets().values().get(
                        spreadsheetId=self.spreadsheet_id,
                        range="1:1"  # First row only
                    ).execute()
                    
                    existing_headers = result.get('values', [[]])[0] if result.get('values') else []
                    
                    # If headers are already correct, don't change anything
                    if existing_headers == expected_headers:
                        return
                        
                except Exception as e:
                    # If we can't read headers, proceed with setting them
                    pass
                
                # Clear and set headers (this will remove all existing data!)
                self.sheets_service.spreadsheets().values().clear(
                    spreadsheetId=self.spreadsheet_id,
                    range="A:Z"
                ).execute()
                
                self.sheets_service.spreadsheets().values().update(
                    spreadsheetId=self.spreadsheet_id,
                    range="A1",
                    valueInputOption="RAW",
                    body={"values": [expected_headers]}
                ).execute()
                
            except Exception as e:
                print(f"Warning: Could not initialize spreadsheet headers: {e}")

    def get_column_mapping(self, model) -> t.Dict:
        """Get mapping between model fields and spreadsheet columns."""
        # For Google Sheets, column names directly match field names (like CSV)
        return model.model_fields

    def load_entries(self, model_class):
        """Load all entries from the Google Sheet."""
        try:
            # Get all data from the sheet
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range="A:Z"
            ).execute()
            
            values = result.get('values', [])
            if not values:
                return []
            
            # First row contains headers
            headers = values[0]
            entries = []
            
            for row_data in values[1:]:  # Skip header row
                try:
                    # Skip empty rows
                    if not row_data or all(not cell.strip() for cell in row_data if cell):
                        continue
                    
                    # Pad row data to match headers length
                    while len(row_data) < len(headers):
                        row_data.append("")
                    
                    # Create dictionary from headers and row data
                    row_dict = dict(zip(headers, row_data))
                    
                    # Extract row_id and remove from model data
                    row_id = row_dict.get("_row_id", str(uuid.uuid4()))
                    
                    # Create model data without _row_id
                    model_data = {k: v for k, v in row_dict.items() if k != "_row_id"}
                    
                    # Skip if all values are empty
                    if not any(str(v).strip() for v in model_data.values()):
                        continue
                    
                    # Convert types as needed
                    typed_row = {}
                    for field, value in model_data.items():
                        if field in model_class.model_fields:
                            field_type = model_class.model_fields[field].annotation
                            
                            # Handle basic type conversions
                            if field_type == int:
                                typed_row[field] = int(value) if value else 0
                            elif field_type == float:
                                typed_row[field] = float(value) if value else 0.0
                            elif field_type == bool:
                                typed_row[field] = value.lower() in ("true", "t", "yes", "y", "1")
                            else:
                                typed_row[field] = value
                    
                    # Create model instance
                    entry = model_class(**typed_row)
                    entry._row_id = row_id
                    entries.append(entry)
                    
                except Exception as e:
                    print(f"Error loading row from Google Sheets: {e}")
            
            return entries
            
        except Exception as e:
            print(f"Error loading entries from Google Sheets: {e}")
            return []

    def append_entry(self, entry):
        """Add a new entry to the Google Sheet and return a generated ID."""
        # Generate row ID if needed
        row_id = getattr(entry, "_row_id", None) or str(uuid.uuid4())
        
        # Convert entry to row data
        entry_dict = entry.model_dump()
        entry_dict["_row_id"] = row_id
        
        # Get headers to maintain order
        headers = ["_row_id"] + list(entry.model_fields.keys())
        row_data = [entry_dict.get(header, "") for header in headers]
        
        # Append to sheet
        self.sheets_service.spreadsheets().values().append(
            spreadsheetId=self.spreadsheet_id,
            range="A:A",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [row_data]}
        ).execute()
        
        return row_id

    def update_entry(self, entry):
        """Update an existing entry in the Google Sheet."""
        if not hasattr(entry, "_row_id") or not entry._row_id:
            raise ValueError("Cannot update: entry has no row ID")
        
        # Find the row with this ID
        result = self.sheets_service.spreadsheets().values().get(
            spreadsheetId=self.spreadsheet_id,
            range="A:A"
        ).execute()
        
        row_ids = [row[0] if row else "" for row in result.get('values', [])]
        
        try:
            row_index = row_ids.index(entry._row_id)
            row_number = row_index + 1  # Sheets are 1-indexed
            
            # Convert entry to row data
            entry_dict = entry.model_dump()
            entry_dict["_row_id"] = entry._row_id
            
            # Get headers to maintain order
            headers = ["_row_id"] + list(entry.model_fields.keys())
            row_data = [entry_dict.get(header, "") for header in headers]
            
            # Update the specific row
            self.sheets_service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=f"A{row_number}:Z{row_number}",
                valueInputOption="RAW",
                body={"values": [row_data]}
            ).execute()
            
            return True
            
        except ValueError:
            # Row ID not found, append as new entry
            return self.append_entry(entry)

    def delete_entry(self, entry_id):
        """Delete an entry from the Google Sheet."""
        # Find the row with this ID
        result = self.sheets_service.spreadsheets().values().get(
            spreadsheetId=self.spreadsheet_id,
            range="A:A"
        ).execute()
        
        row_ids = [row[0] if row else "" for row in result.get('values', [])]
        
        try:
            row_index = row_ids.index(entry_id)
            row_number = row_index + 1  # Sheets are 1-indexed
            
            # Delete the row
            request = {
                'deleteDimension': {
                    'range': {
                        'sheetId': 0,  # Assume first sheet
                        'dimension': 'ROWS',
                        'startIndex': row_index,
                        'endIndex': row_index + 1
                    }
                }
            }
            
            self.sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={'requests': [request]}
            ).execute()
            
            return True
            
        except ValueError:
            # Row ID not found
            return False

    def get_entry_by_field(self, field_name: str, field_value: t.Any, model_class):
        """Get an entry by field value."""
        # Load all entries and search
        entries = self.load_entries(model_class)
        
        for entry in entries:
            if hasattr(entry, field_name) and getattr(entry, field_name) == field_value:
                return entry
        
        return None
