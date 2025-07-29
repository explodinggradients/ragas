"""Tests for Google Drive backend implementation."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel

try:
    from googleapiclient.errors import HttpError

    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

    # Create a mock HttpError for testing when Google API isn't available
    class HttpError(Exception):
        def __init__(self, resp, content):
            self.resp = resp
            self.content = content
            super().__init__()


from ragas_experimental.backends.gdrive_backend import GDriveBackend, GDRIVE_AVAILABLE


class SampleModel(BaseModel):
    name: str
    value: int
    description: str


class TestGDriveBackendAvailability:
    """Test Google Drive backend availability and import handling."""

    def test_gdrive_available_import(self):
        """Test that GDRIVE_AVAILABLE reflects actual import capability."""
        # This test will pass if the Google Drive dependencies are installed
        # and fail gracefully if they're not
        if GDRIVE_AVAILABLE:
            # If available, we should be able to create the backend class
            assert GDriveBackend is not None
        else:
            # If not available, importing should have failed gracefully
            pytest.skip("Google Drive dependencies not available")


@pytest.mark.skipif(
    not GDRIVE_AVAILABLE, reason="Google Drive dependencies not available"
)
class TestGDriveBackendInitialization:
    """Test GDriveBackend initialization and authentication setup."""

    @patch("ragas_experimental.backends.gdrive_backend.build")
    @patch("ragas_experimental.backends.gdrive_backend.Credentials")
    @patch("os.path.exists")
    def test_service_account_auth_success(
        self, mock_exists, mock_credentials, mock_build
    ):
        """Test successful service account authentication."""
        mock_exists.return_value = True
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds
        mock_drive_service = Mock()
        mock_sheets_service = Mock()
        mock_build.side_effect = [mock_drive_service, mock_sheets_service]

        # Mock the folder structure setup
        mock_drive_service.files().get.return_value.execute.return_value = {
            "id": "test_folder"
        }
        mock_drive_service.files().list.return_value.execute.side_effect = [
            {"files": []},
            {"files": []},  # No existing folders
        ]
        mock_drive_service.files().create.return_value.execute.side_effect = [
            {"id": "datasets_folder"},
            {"id": "experiments_folder"},
        ]

        backend = GDriveBackend(
            folder_id="test_folder",
            service_account_path="/path/to/service_account.json",
        )

        assert backend.folder_id == "test_folder"
        assert backend.drive_service == mock_drive_service
        assert backend.sheets_service == mock_sheets_service

        mock_credentials.from_service_account_file.assert_called_once()

    @patch("ragas_experimental.backends.gdrive_backend.build")
    @patch("os.path.exists")
    def test_auth_failure_no_credentials(self, mock_exists, mock_build):
        """Test authentication failure when no credentials are provided."""
        mock_exists.return_value = False

        with pytest.raises(ValueError, match="No valid authentication method found"):
            GDriveBackend(folder_id="test_folder")

    @patch("ragas_experimental.backends.gdrive_backend.build")
    @patch("ragas_experimental.backends.gdrive_backend.Credentials")
    @patch("os.path.exists")
    def test_invalid_folder_id(self, mock_exists, mock_credentials, mock_build):
        """Test behavior with invalid folder ID."""
        mock_exists.return_value = True
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds
        mock_drive_service = Mock()
        mock_sheets_service = Mock()
        mock_build.side_effect = [mock_drive_service, mock_sheets_service]

        # Mock folder not found with specific Google API error
        mock_response = Mock()
        mock_response.status = 404
        mock_drive_service.files().get.side_effect = HttpError(
            mock_response, b'{"error": {"message": "File not found"}}'
        )

        with pytest.raises(ValueError, match="Folder with ID test_folder not found"):
            GDriveBackend(
                folder_id="test_folder",
                service_account_path="/path/to/service_account.json",
            )


@pytest.mark.skipif(
    not GDRIVE_AVAILABLE, reason="Google Drive dependencies not available"
)
class TestGDriveBackendOperations:
    """Test Google Drive backend data operations."""

    def _create_mock_backend(self):
        """Helper to create a mocked GDriveBackend instance."""
        with patch("ragas_experimental.backends.gdrive_backend.build"):
            with patch("ragas_experimental.backends.gdrive_backend.Credentials"):
                with patch("os.path.exists", return_value=True):
                    backend = GDriveBackend(
                        folder_id="test_folder", service_account_path="/fake/path.json"
                    )
                    # Mock the required folder IDs
                    backend.datasets_folder_id = "datasets_folder"
                    backend.experiments_folder_id = "experiments_folder"
                    return backend

    def test_spreadsheet_exists_check(self):
        """Test checking if a spreadsheet exists."""
        backend = self._create_mock_backend()

        # Mock existing spreadsheet
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": [{"id": "existing_spreadsheet"}]
        }

        assert backend._spreadsheet_exists("test_dataset", "datasets") is True

        # Mock non-existing spreadsheet
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": []
        }

        assert backend._spreadsheet_exists("nonexistent", "datasets") is False

    def test_load_nonexistent_dataset(self):
        """Test loading a dataset that doesn't exist."""
        backend = self._create_mock_backend()

        # Mock non-existing spreadsheet
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": []
        }

        with pytest.raises(FileNotFoundError, match="Dataset 'nonexistent' not found"):
            backend.load_dataset("nonexistent")

    def test_load_dataset_success(self):
        """Test successful dataset loading."""
        backend = self._create_mock_backend()

        # Mock existing spreadsheet
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": [{"id": "test_spreadsheet"}]
        }

        # Mock spreadsheet data
        mock_data = {
            "values": [
                ["name", "value", "description"],  # Headers
                ["Item 1", "10", "First item"],
                ["Item 2", "20", "Second item"],
            ]
        }
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = (
            mock_data
        )

        result = backend.load_dataset("test_dataset")

        assert len(result) == 2
        assert result[0]["name"] == "Item 1"
        assert result[0]["value"] == 10  # Should be converted to int
        assert result[1]["name"] == "Item 2"
        assert result[1]["value"] == 20

    def test_load_empty_dataset(self):
        """Test loading an empty dataset."""
        backend = self._create_mock_backend()

        # Mock existing but empty spreadsheet
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": [{"id": "test_spreadsheet"}]
        }
        backend.sheets_service.spreadsheets().values().get.return_value.execute.return_value = {
            "values": []
        }

        result = backend.load_dataset("empty_dataset")
        assert result == []

    def test_save_dataset_success(self):
        """Test successful dataset saving."""
        backend = self._create_mock_backend()

        # Mock spreadsheet creation
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": []
        }
        backend.drive_service.files().create.return_value.execute.return_value = {
            "id": "new_spreadsheet"
        }

        # Mock sheets operations
        backend.sheets_service.spreadsheets().values().clear.return_value.execute.return_value = (
            {}
        )
        backend.sheets_service.spreadsheets().values().update.return_value.execute.return_value = (
            {}
        )

        test_data = [
            {"name": "Test Item", "value": 42, "description": "Test description"}
        ]

        # Should not raise any exceptions
        backend.save_dataset("test_dataset", test_data)

        # Verify the update was called
        backend.sheets_service.spreadsheets().values().update.assert_called_once()

    def test_save_empty_dataset(self):
        """Test saving an empty dataset."""
        backend = self._create_mock_backend()

        # Mock existing spreadsheet
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": [{"id": "test_spreadsheet"}]
        }
        backend.sheets_service.spreadsheets().values().clear.return_value.execute.return_value = (
            {}
        )

        # Should clear the spreadsheet
        backend.save_dataset("empty_dataset", [])

        # Verify clear was called
        backend.sheets_service.spreadsheets().values().clear.assert_called_once()

    def test_list_datasets(self):
        """Test listing available datasets."""
        backend = self._create_mock_backend()

        # Mock spreadsheets in the datasets folder (only spreadsheets should be returned by the API query)
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": [
                {
                    "name": "dataset1.gsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                {
                    "name": "dataset2.gsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
            ]
        }

        result = backend.list_datasets()

        assert sorted(result) == ["dataset1", "dataset2"]

    def test_list_experiments(self):
        """Test listing available experiments."""
        backend = self._create_mock_backend()

        # Mock spreadsheets in the experiments folder
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": [{"name": "experiment1.gsheet"}, {"name": "experiment2.gsheet"}]
        }

        result = backend.list_experiments()

        assert sorted(result) == ["experiment1", "experiment2"]

    def test_complex_data_serialization(self):
        """Test that complex data (lists, dicts) gets JSON serialized."""
        backend = self._create_mock_backend()

        # Mock spreadsheet creation
        backend.drive_service.files().list.return_value.execute.return_value = {
            "files": []
        }
        backend.drive_service.files().create.return_value.execute.return_value = {
            "id": "new_spreadsheet"
        }

        # Capture the data that gets sent to the sheets API
        mock_update = Mock()
        backend.sheets_service.spreadsheets().values().update.return_value.execute = (
            mock_update
        )
        backend.sheets_service.spreadsheets().values().clear.return_value.execute.return_value = (
            {}
        )

        test_data = [
            {
                "name": "Test",
                "complex_list": [1, 2, 3],
                "complex_dict": {"nested": "value"},
            }
        ]

        backend.save_dataset("complex_dataset", test_data)

        # Verify update was called and check the serialization
        backend.sheets_service.spreadsheets().values().update.assert_called_once()
        call_args = backend.sheets_service.spreadsheets().values().update.call_args
        sheet_data = call_args[1]["body"]["values"]

        # Should have headers + 1 data row
        assert len(sheet_data) == 2
        # Check that complex data was JSON serialized
        data_row = sheet_data[1]
        assert "[1, 2, 3]" in data_row  # List serialized
        assert '{"nested": "value"}' in data_row  # Dict serialized


@pytest.mark.skipif(
    not GDRIVE_AVAILABLE, reason="Google Drive dependencies not available"
)
class TestGDriveBackendIntegration:
    """Test integration aspects of the Google Drive backend."""

    def test_backend_implements_basebackend(self):
        """Test that GDriveBackend properly implements BaseBackend interface."""
        from ragas_experimental.backends.base import BaseBackend

        assert issubclass(GDriveBackend, BaseBackend)

        # Check that all required methods are implemented
        required_methods = [
            "load_dataset",
            "load_experiment",
            "save_dataset",
            "save_experiment",
            "list_datasets",
            "list_experiments",
        ]

        for method in required_methods:
            assert hasattr(GDriveBackend, method)
            assert callable(getattr(GDriveBackend, method))

    def test_error_without_dependencies(self):
        """Test error handling when Google Drive dependencies are missing."""
        # This test simulates the case where dependencies are not installed
        with patch(
            "ragas_experimental.backends.gdrive_backend.GDRIVE_AVAILABLE", False
        ):
            # Should raise ImportError when trying to create backend
            with pytest.raises(
                ImportError,
                match="Google Drive backend requires additional dependencies",
            ):
                GDriveBackend(folder_id="test")


if __name__ == "__main__":
    pytest.main([__file__])
