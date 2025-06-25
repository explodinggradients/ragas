"""
Unit tests for Google Drive backend implementation.
These tests comprehensively test the GDriveBackend class with proper mocking.
"""

import pytest
import uuid
import json
import sys
from unittest.mock import Mock, patch, MagicMock, mock_open
from pydantic import BaseModel

from ragas_experimental.typing import SUPPORTED_BACKENDS
from ragas_experimental.dataset import create_dataset_backend
from ragas_experimental.project.core import Project


class SampleModel(BaseModel):
    name: str
    value: int
    description: str


class TestGDriveBackendSupport:
    """Test Google Drive backend support in the system."""
    
    def test_gdrive_backend_in_supported_backends(self):
        """Test that gdrive is included in supported backends."""
        assert "gdrive" in SUPPORTED_BACKENDS.__args__

    def test_create_dataset_backend_handles_gdrive_missing_deps(self):
        """Test that factory handles missing Google Drive dependencies gracefully."""
        # Temporarily modify GDRIVE_AVAILABLE to simulate missing dependencies
        from ragas_experimental import dataset
        original_gdrive_available = dataset.GDRIVE_AVAILABLE
        original_gdrive_backend = dataset.GDriveBackend
        
        try:
            # Simulate missing dependencies
            dataset.GDRIVE_AVAILABLE = False
            dataset.GDriveBackend = None
            
            with pytest.raises(ImportError, match="Google Drive backend requires additional dependencies"):
                create_dataset_backend(
                    "gdrive",
                    folder_id="test_folder",
                    project_id="test_project",
                    dataset_id="test_dataset",
                    dataset_name="test_dataset",
                    type="datasets"
                )
        finally:
            # Restore original values
            dataset.GDRIVE_AVAILABLE = original_gdrive_available
            dataset.GDriveBackend = original_gdrive_backend


class TestGDriveBackendInitialization:
    """Test GDriveBackend initialization and authentication setup."""

    @patch('ragas_experimental.backends.gdrive_backend.build')
    @patch('ragas_experimental.backends.gdrive_backend.Credentials')
    @patch('os.path.exists')
    def test_service_account_auth_success(self, mock_exists, mock_credentials, mock_build):
        """Test successful service account authentication."""
        mock_exists.return_value = True
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds
        mock_drive_service = Mock()
        mock_sheets_service = Mock()
        mock_build.side_effect = [mock_drive_service, mock_sheets_service]
        
        try:
            from ragas_experimental.backends.gdrive_backend import GDriveBackend
            
            backend = GDriveBackend(
                folder_id="test_folder",
                project_id="test_project",
                dataset_id="test_dataset",
                dataset_name="test_dataset",
                type="datasets",
                service_account_path="/path/to/service_account.json"
            )
            
            assert backend.folder_id == "test_folder"
            assert backend.project_id == "test_project"
            assert backend.dataset_id == "test_dataset"
            assert backend.dataset_name == "test_dataset"
            assert backend.type == "datasets"
            assert backend.drive_service == mock_drive_service
            assert backend.sheets_service == mock_sheets_service
            
            mock_credentials.from_service_account_file.assert_called_once_with(
                "/path/to/service_account.json", 
                scopes=['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
            )
            
        except ImportError:
            pytest.skip("Google Drive dependencies not available")

    @patch('ragas_experimental.backends.gdrive_backend.build')
    @patch('ragas_experimental.backends.gdrive_backend.UserCredentials')
    @patch('ragas_experimental.backends.gdrive_backend.InstalledAppFlow')
    @patch('os.path.exists')
    def test_oauth_auth_new_token(self, mock_exists, mock_flow, mock_user_creds, mock_build):
        """Test OAuth authentication with new token creation."""
        # Mock file existence checks
        def exists_side_effect(path):
            return path == "/path/to/credentials.json"
        mock_exists.side_effect = exists_side_effect
        
        # Mock OAuth flow
        mock_flow_instance = Mock()
        mock_flow.from_client_secrets_file.return_value = mock_flow_instance
        mock_creds = Mock()
        mock_creds.valid = True
        mock_flow_instance.run_local_server.return_value = mock_creds
        mock_creds.to_json.return_value = '{"token": "test"}'
        
        try:
            from ragas_experimental.backends.gdrive_backend import GDriveBackend
            
            with patch("builtins.open", mock_open()) as mock_file:
                backend = GDriveBackend(
                    folder_id="test_folder",
                    project_id="test_project",
                    dataset_id="test_dataset",
                    dataset_name="test_dataset",
                    type="datasets",
                    credentials_path="/path/to/credentials.json",
                    token_path="/path/to/token.json"
                )
                
                # Verify OAuth flow was initiated
                mock_flow.from_client_secrets_file.assert_called_once()
                mock_flow_instance.run_local_server.assert_called_once_with(port=0)
                mock_file.assert_called_with("/path/to/token.json", 'w')
                
        except ImportError:
            pytest.skip("Google Drive dependencies not available")

    @patch('ragas_experimental.backends.gdrive_backend.build')
    @patch('os.path.exists')
    def test_auth_failure_no_credentials(self, mock_exists, mock_build):
        """Test authentication failure when no credentials are provided."""
        mock_exists.return_value = False
        
        try:
            from ragas_experimental.backends.gdrive_backend import GDriveBackend
            
            with pytest.raises(ValueError, match="No valid authentication method found"):
                GDriveBackend(
                    folder_id="test_folder",
                    project_id="test_project",
                    dataset_id="test_dataset",
                    dataset_name="test_dataset",
                    type="datasets"
                )
                
        except ImportError:
            pytest.skip("Google Drive dependencies not available")


class TestGDriveBackendFolderManagement:
    """Test folder structure management in Google Drive."""

    def _create_mock_backend(self):
        """Helper to create a mocked GDriveBackend instance."""
        with patch('ragas_experimental.backends.gdrive_backend.build'):
            with patch('ragas_experimental.backends.gdrive_backend.Credentials'):
                with patch('os.path.exists', return_value=True):
                    try:
                        from ragas_experimental.backends.gdrive_backend import GDriveBackend
                        return GDriveBackend(
                            folder_id="test_folder",
                            project_id="test_project",
                            dataset_id="test_dataset",
                            dataset_name="test_dataset",
                            type="datasets",
                            service_account_path="/fake/path.json"
                        )
                    except ImportError:
                        pytest.skip("Google Drive dependencies not available")

    def test_ensure_folder_structure_success(self):
        """Test successful folder structure creation."""
        backend = self._create_mock_backend()
        if backend is None:
            return
            
        # Mock successful folder operations
        backend.drive_service.files().get.return_value.execute.return_value = {"id": "main_folder"}
        backend.drive_service.files().list.return_value.execute.side_effect = [
            {"files": []},  # No project folder exists
            {"files": []},  # No type folder exists
        ]
        backend.drive_service.files().create.return_value.execute.side_effect = [
            {"id": "project_folder_id"},
            {"id": "type_folder_id"}
        ]
        
        backend._ensure_folder_structure()
        
        assert backend.type_folder_id == "type_folder_id"
        assert backend.drive_service.files().create.call_count == 2

    def test_ensure_folder_structure_existing_folders(self):
        """Test folder structure with existing folders."""
        backend = self._create_mock_backend()
        if backend is None:
            return
            
        backend.drive_service.files().get.return_value.execute.return_value = {"id": "main_folder"}
        backend.drive_service.files().list.return_value.execute.side_effect = [
            {"files": [{"id": "existing_project_folder"}]},  # Project folder exists
            {"files": [{"id": "existing_type_folder"}]},     # Type folder exists
        ]
        
        backend._ensure_folder_structure()
        
        assert backend.type_folder_id == "existing_type_folder"
        backend.drive_service.files().create.assert_not_called()

    def test_ensure_folder_structure_invalid_main_folder(self):
        """Test folder structure with invalid main folder."""
        backend = self._create_mock_backend()
        if backend is None:
            return
            
        backend.drive_service.files().get.side_effect = Exception("Not found")
        
        with pytest.raises(ValueError, match="Folder with ID test_folder not found"):
            backend._ensure_folder_structure()


class TestGDriveBackendSpreadsheetManagement:
    """Test spreadsheet management operations."""

    def _create_mock_backend_with_dataset(self):
        """Helper to create a mocked backend with a dataset."""
        backend = TestGDriveBackendFolderManagement()._create_mock_backend()
        if backend is None:
            return None
            
        # Mock a simple dataset
        mock_dataset = Mock()
        mock_dataset.model = SampleModel
        backend.dataset = mock_dataset
        backend.type_folder_id = "type_folder_id"
        
        return backend

    def test_ensure_spreadsheet_create_new(self):
        """Test creating a new spreadsheet."""
        backend = self._create_mock_backend_with_dataset()
        if backend is None:
            return
            
        # Mock no existing spreadsheet
        backend.drive_service.files().list.return_value.execute.return_value = {"files": []}
        backend.drive_service.files().create.return_value.execute.return_value = {"id": "new_spreadsheet_id"}
        
        # Mock sheets service for header initialization
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = {"values": []}
        backend.sheets_service.spreadsheets().values().clear.return_value.execute.return_value = {}
        backend.sheets_service.spreadsheets().values().update.return_value.execute.return_value = {}
        
        backend._ensure_spreadsheet_exists()
        
        assert backend.spreadsheet_id == "new_spreadsheet_id"
        backend.drive_service.files().create.assert_called_once()

    def test_ensure_spreadsheet_use_existing(self):
        """Test using existing spreadsheet."""
        backend = self._create_mock_backend_with_dataset()
        if backend is None:
            return
            
        # Mock existing spreadsheet
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": [{"id": "existing_spreadsheet_id"}]
        }
        
        # Mock existing headers that match expected
        expected_headers = ["_row_id", "name", "value", "description"]
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = {
            "values": [expected_headers]
        }
        
        backend._ensure_spreadsheet_exists()
        
        assert backend.spreadsheet_id == "existing_spreadsheet_id"
        backend.drive_service.files().create.assert_not_called()

    def test_initialize_spreadsheet_headers(self):
        """Test spreadsheet header initialization."""
        backend = self._create_mock_backend_with_dataset()
        if backend is None:
            return
            
        backend.spreadsheet_id = "test_spreadsheet"
        
        # Mock no existing headers
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = {"values": []}
        backend.sheets_service.spreadsheets().values().clear.return_value.execute.return_value = {}
        backend.sheets_service.spreadsheets().values().update.return_value.execute.return_value = {}
        
        backend._initialize_spreadsheet_headers()
        
        # Verify headers were set
        expected_headers = ["_row_id", "name", "value", "description"]
        backend.sheets_service.spreadsheets().values().update.assert_called_once()
        call_args = backend.sheets_service.spreadsheets().values().update.call_args
        assert call_args[1]["body"]["values"][0] == expected_headers


class TestGDriveBackendDataOperations:
    """Test data operations (CRUD) on spreadsheets."""

    def _create_mock_backend_with_spreadsheet(self):
        """Helper to create a mocked backend with spreadsheet setup."""
        backend = TestGDriveBackendFolderManagement()._create_mock_backend()
        if backend is None:
            return None
            
        backend.spreadsheet_id = "test_spreadsheet"
        return backend

    def test_load_entries_success(self):
        """Test successful loading of entries from spreadsheet."""
        backend = self._create_mock_backend_with_spreadsheet()
        if backend is None:
            return
            
        # Mock spreadsheet data
        mock_data = {
            "values": [
                ["_row_id", "name", "value", "description"],  # Headers
                ["row1", "Item 1", "10", "First item"],
                ["row2", "Item 2", "20", "Second item"],
                ["", "", "", ""],  # Empty row should be skipped
            ]
        }
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = mock_data
        
        entries = backend.load_entries(SampleModel)
        
        assert len(entries) == 2
        assert entries[0].name == "Item 1"
        assert entries[0].value == 10
        assert entries[0]._row_id == "row1"
        assert entries[1].name == "Item 2"
        assert entries[1].value == 20

    def test_load_entries_empty_spreadsheet(self):
        """Test loading entries from empty spreadsheet."""
        backend = self._create_mock_backend_with_spreadsheet()
        if backend is None:
            return
            
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = {"values": []}
        
        entries = backend.load_entries(SampleModel)
        
        assert entries == []

    def test_append_entry_success(self):
        """Test successful entry appending."""
        backend = self._create_mock_backend_with_spreadsheet()
        if backend is None:
            return
            
        backend.sheets_service.spreadsheets().values().append.return_value.execute.return_value = {}
        
        entry = SampleModel(name="New Item", value=30, description="New description")
        row_id = backend.append_entry(entry)
        
        assert isinstance(row_id, str)
        assert len(row_id) > 0
        backend.sheets_service.spreadsheets().values().append.assert_called_once()

    def test_update_entry_success(self):
        """Test successful entry update."""
        backend = self._create_mock_backend_with_spreadsheet()
        if backend is None:
            return
            
        # Mock finding the row
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = {
            "values": [["existing_row_id"], ["other_row_id"]]
        }
        backend.sheets_service.spreadsheets().values().update.return_value.execute.return_value = {}
        
        entry = SampleModel(name="Updated Item", value=40, description="Updated description")
        entry._row_id = "existing_row_id"
        
        result = backend.update_entry(entry)
        
        assert result is True
        backend.sheets_service.spreadsheets().values().update.assert_called_once()

    def test_update_entry_not_found_appends(self):
        """Test update entry creates new entry when row not found."""
        backend = self._create_mock_backend_with_spreadsheet()
        if backend is None:
            return
            
        # Mock not finding the row
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = {
            "values": [["other_row_id"]]
        }
        backend.sheets_service.spreadsheets().values().append.return_value.execute.return_value = {}
        
        entry = SampleModel(name="New Item", value=50, description="New description")
        entry._row_id = "nonexistent_row_id"
        
        result = backend.update_entry(entry)
        
        # Should append since row not found
        backend.sheets_service.spreadsheets().values().append.assert_called_once()

    def test_delete_entry_success(self):
        """Test successful entry deletion."""
        backend = self._create_mock_backend_with_spreadsheet()
        if backend is None:
            return
            
        # Mock finding the row
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = {
            "values": [["row_to_delete"], ["other_row"]]
        }
        backend.sheets_service.spreadsheets().batchUpdate.return_value.execute.return_value = {}
        
        result = backend.delete_entry("row_to_delete")
        
        assert result is True
        backend.sheets_service.spreadsheets().batchUpdate.assert_called_once()

    def test_delete_entry_not_found(self):
        """Test delete entry when row not found."""
        backend = self._create_mock_backend_with_spreadsheet()
        if backend is None:
            return
            
        # Mock not finding the row
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = {
            "values": [["other_row"]]
        }
        
        result = backend.delete_entry("nonexistent_row")
        
        assert result is False
        backend.sheets_service.spreadsheets().batchUpdate.assert_not_called()

    def test_get_entry_by_field_success(self):
        """Test finding entry by field value."""
        backend = self._create_mock_backend_with_spreadsheet()
        if backend is None:
            return
            
        # Mock load_entries to return test data
        mock_entries = [
            SampleModel(name="Item 1", value=10, description="First item"),
            SampleModel(name="Item 2", value=20, description="Second item"),
        ]
        mock_entries[0]._row_id = "row1"
        mock_entries[1]._row_id = "row2"
        
        with patch.object(backend, 'load_entries', return_value=mock_entries):
            entry = backend.get_entry_by_field("name", "Item 1", SampleModel)
            
            assert entry is not None
            assert entry.name == "Item 1"
            assert entry.value == 10

    def test_get_entry_by_field_not_found(self):
        """Test finding entry by field value when not found."""
        backend = self._create_mock_backend_with_spreadsheet()
        if backend is None:
            return
            
        with patch.object(backend, 'load_entries', return_value=[]):
            entry = backend.get_entry_by_field("name", "Nonexistent", SampleModel)
            
            assert entry is None


class TestGDriveBackendIntegration:
    """Test integration with Project and Dataset classes."""

    def test_project_create_gdrive_params(self):
        """Test that Project.create accepts Google Drive parameters."""
        try:
            with patch('ragas_experimental.backends.gdrive_backend.build'):
                with patch('ragas_experimental.backends.gdrive_backend.Credentials'):
                    with patch('os.path.exists', return_value=True):
                        project = Project.create(
                            name="test_project",
                            backend="gdrive",
                            gdrive_folder_id="test_folder",
                            gdrive_service_account_path="fake_path.json"
                        )
                        assert project.backend == "gdrive"
                        assert project._gdrive_folder_id == "test_folder"
        except ImportError:
            pytest.skip("Google Drive dependencies not available")

    def test_project_gdrive_validation(self):
        """Test that Project validates required Google Drive parameters."""
        with pytest.raises(ValueError, match="gdrive_folder_id is required"):
            Project.create(
                name="test_project",
                backend="gdrive"
                # Missing gdrive_folder_id
            )

    def test_get_column_mapping(self):
        """Test get_column_mapping returns model fields."""
        backend = TestGDriveBackendFolderManagement()._create_mock_backend()
        if backend is None:
            return
            
        mapping = backend.get_column_mapping(SampleModel)
        
        # Should return the model fields directly
        assert mapping == SampleModel.model_fields

    def test_str_and_repr(self):
        """Test string representations of backend."""
        backend = TestGDriveBackendFolderManagement()._create_mock_backend()
        if backend is None:
            return
            
        str_repr = str(backend)
        assert "GDriveBackend" in str_repr
        assert "test_folder" in str_repr
        assert "test_project" in str_repr
        
        assert str(backend) == repr(backend)


if __name__ == "__main__":
    pytest.main([__file__])
