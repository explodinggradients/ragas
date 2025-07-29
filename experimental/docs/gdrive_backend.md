# Google Drive Backend for Ragas

The Google Drive backend allows you to store Ragas datasets and experiments in Google Sheets within your Google Drive. This provides a cloud-based, collaborative storage solution that's familiar to many users.

## Features

- **Cloud Storage**: Store your datasets and experiments in Google Drive
- **Collaborative**: Share and collaborate on datasets using Google Drive's sharing features
- **Google Sheets Format**: Data is stored in Google Sheets for easy viewing and editing
- **Automatic Structure**: Creates organized folder structure (datasets/ and experiments/)
- **Type Preservation**: Attempts to preserve basic data types (strings, numbers)
- **Multiple Authentication**: Supports both OAuth and Service Account authentication

## Installation

```bash
# Install with Google Drive dependencies
pip install "ragas_experimental[gdrive]"
```

## Setup

### 1. Google Cloud Project Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Google Drive API
   - Google Sheets API

### 2. Authentication Setup

Choose one of two authentication methods:

#### Option A: Service Account (Recommended)

1. In Google Cloud Console, go to "Credentials"
2. Click "Create Credentials" → "Service account"
3. Create the service account and download the JSON key file
4. Share your Google Drive folder with the service account email

*This is the preferred method as it works well for both scripts and production environments without requiring user interaction.*

#### Option B: OAuth 2.0 (Alternative for Interactive Use)

1. In Google Cloud Console, go to "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. Choose "Desktop application"
4. Download the JSON file (save as `credentials.json`)

### 3. Google Drive Folder Setup

1. Create a folder in Google Drive for your Ragas data
2. Get the folder ID from the URL: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
3. If using Service Account, share the folder with the service account email

## Usage

### Basic Usage

```python
from ragas_experimental.dataset import Dataset
from pydantic import BaseModel

# Define your data model
class EvaluationRecord(BaseModel):
    question: str
    answer: str
    score: float

# Create dataset with Google Drive backend
dataset = Dataset(
    name="my_evaluation",
    backend="gdrive",
    data_model=EvaluationRecord,
    folder_id="your_google_drive_folder_id",
    credentials_path="path/to/credentials.json"
)

# Add data
record = EvaluationRecord(
    question="What is AI?",
    answer="Artificial Intelligence",
    score=0.95
)
dataset.append(record)

# Save to Google Drive
dataset.save()

# Load from Google Drive
dataset.load()
```

### Authentication Options

#### Using Environment Variables

```bash
export GDRIVE_FOLDER_ID="your_folder_id"
export GDRIVE_CREDENTIALS_PATH="path/to/credentials.json"
# OR for service account:
export GDRIVE_SERVICE_ACCOUNT_PATH="path/to/service_account.json"
```

```python
# Environment variables will be used automatically
dataset = Dataset(
    name="my_evaluation",
    backend="gdrive",
    data_model=EvaluationRecord,
    folder_id=os.getenv("GDRIVE_FOLDER_ID")
)
```

#### Using Service Account

```python
dataset = Dataset(
    name="my_evaluation",
    backend="gdrive",
    data_model=EvaluationRecord,
    folder_id="your_folder_id",
    service_account_path="path/to/service_account.json"
)
```

#### Custom Token Path

```python
dataset = Dataset(
    name="my_evaluation",
    backend="gdrive",
    data_model=EvaluationRecord,
    folder_id="your_folder_id",
    credentials_path="path/to/credentials.json",
    token_path="custom_token.json"
)
```

## File Structure

The backend creates the following structure in your Google Drive folder:

```text
Your Google Drive Folder/
├── datasets/
│   ├── dataset1.gsheet
│   ├── dataset2.gsheet
│   └── ...
└── experiments/
    ├── experiment1.gsheet
    ├── experiment2.gsheet
    └── ...
```

Each dataset/experiment is stored as a separate Google Sheet with:

- Column headers matching your data model fields
- Automatic type conversion for basic types (int, float, string)
- JSON serialization for complex objects

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GDRIVE_FOLDER_ID` | Google Drive folder ID | `1abc123...` |
| `GDRIVE_CREDENTIALS_PATH` | Path to OAuth credentials JSON | `./credentials.json` |
| `GDRIVE_SERVICE_ACCOUNT_PATH` | Path to service account JSON | `./service_account.json` |
| `GDRIVE_TOKEN_PATH` | Path to store OAuth token | `./token.json` |

## Best Practices

### Security

- Never commit credential files to version control
- Use environment variables for sensitive information
- Regularly rotate service account keys
- Use OAuth for development, service accounts for production

### Performance

- Google Sheets API has rate limits - avoid frequent saves with large datasets
- Consider batching operations when possible
- Use appropriate folder organization for large numbers of datasets

### Collaboration

- Share folders with appropriate permissions (view/edit)
- Use descriptive dataset names
- Document your data models clearly

## Troubleshooting

### Common Issues

1. **"Folder not found" error**
   - Verify the folder ID is correct
   - Ensure the folder is shared with your service account (if using one)
   - Check that the folder exists and is accessible

2. **Authentication errors**
   - Verify credential file paths are correct
   - Check that required APIs are enabled in Google Cloud Console
   - For OAuth: delete token file and re-authenticate
   - For Service Account: verify the JSON file is valid

3. **Permission errors**
   - Ensure your account has edit access to the folder
   - For service accounts: share the folder with the service account email
   - Check Google Drive sharing settings

4. **Import errors**
   - Install dependencies: `pip install "ragas_experimental[gdrive]"`
   - Verify all required packages are installed

### Getting Help

If you encounter issues:

1. Check error messages carefully for specific details
2. Verify your Google Cloud project setup
3. Test with a simple example first
4. Check the Google Drive API documentation for rate limits

## Limitations

- Google Sheets has a limit of 10 million cells per spreadsheet
- Complex nested objects are JSON-serialized as strings
- API rate limits may affect performance with large datasets
- Requires internet connection for all operations

## Examples

See `examples/gdrive_backend_example.py` for a complete working example.
