# Google Drive Backend Setup Guide

This guide will help you set up and use the Google Drive backend for Ragas datasets.

## Prerequisites

### 1. Install Dependencies

```bash
pip install google-api-python-client google-auth google-auth-oauthlib
```

### 2. Set up Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Google Drive API
   - Google Sheets API

### 3. Create Credentials

You have two options for authentication:

#### Option A: OAuth 2.0 (Recommended for development)

1. In Google Cloud Console, go to "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. Choose "Desktop application"
4. Download the JSON file
5. Save it securely (e.g., as `credentials.json`)

#### Option B: Service Account (Recommended for production)

1. In Google Cloud Console, go to "Credentials"
2. Click "Create Credentials" → "Service account"
3. Fill in the details and create the account
4. Generate a key (JSON format)
5. Download and save the JSON file securely
6. Share your Google Drive folder with the service account email

## Setup Instructions

### 1. Create a Google Drive Folder

1. Create a folder in Google Drive where you want to store your datasets
2. Get the folder ID from the URL:
   ```
   https://drive.google.com/drive/folders/FOLDER_ID_HERE
   ```
3. If using a service account, share this folder with the service account email

### 2. Set Environment Variables (Optional)

```bash
export GDRIVE_FOLDER_ID="your_folder_id_here"
export GDRIVE_CREDENTIALS_PATH="path/to/credentials.json"
# OR for service account:
export GDRIVE_SERVICE_ACCOUNT_PATH="path/to/service_account.json"
```

### 3. Basic Usage

```python
from ragas_experimental.project.core import Project
from pydantic import BaseModel

# Define your data model
class EvaluationEntry(BaseModel):
    question: str
    answer: str
    score: float

# Create project with Google Drive backend
project = Project.create(
    name="my_project",
    backend="gdrive",
    gdrive_folder_id="your_folder_id_here",
    gdrive_credentials_path="path/to/credentials.json"  # OAuth
    # OR
    # gdrive_service_account_path="path/to/service_account.json"  # Service Account
)

# Create a dataset
dataset = project.create_dataset(
    model=EvaluationEntry,
    name="my_dataset"
)

# Add data
entry = EvaluationEntry(
    question="What is AI?",
    answer="Artificial Intelligence",
    score=0.95
)
dataset.append(entry)

# Load and access data
dataset.load()
print(f"Dataset has {len(dataset)} entries")
for entry in dataset:
    print(f"{entry.question} -> {entry.answer}")
```

## File Structure

When you use the Google Drive backend, it creates the following structure:

```
Your Google Drive Folder/
├── project_name/
│   ├── datasets/
│   │   ├── dataset1.gsheet
│   │   └── dataset2.gsheet
│   └── experiments/
│       └── experiment1.gsheet
```

Each dataset is stored as a Google Sheet with:
- Column headers matching your model fields
- An additional `_row_id` column for internal tracking
- Automatic type conversion when loading data

## Authentication Flow

### OAuth (First Time)
1. When you first run your code, a browser window will open
2. Sign in to your Google account
3. Grant permissions to access Google Drive
4. A `token.json` file will be created automatically
5. Subsequent runs will use this token (no browser needed)

### Service Account
1. No interactive authentication required
2. Make sure the service account has access to your folder
3. The JSON key file is used directly

## Troubleshooting

### Common Issues

1. **"Folder not found" error**
   - Check that the folder ID is correct
   - Ensure the folder is shared with your service account (if using one)

2. **Authentication errors**
   - Verify your credentials file path
   - Check that the required APIs are enabled
   - For OAuth: Delete `token.json` and re-authenticate

3. **Permission errors**
   - Make sure your account has edit access to the folder
   - For service accounts: share the folder with the service account email

4. **Import errors**
   - Install required dependencies: `pip install google-api-python-client google-auth google-auth-oauthlib`

### Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify your Google Cloud setup
3. Test authentication with a simple Google Drive API call
4. Check that all dependencies are installed

## Security Best Practices

1. **Never commit credentials to version control**
2. **Use environment variables for sensitive information**
3. **Limit service account permissions to minimum required**
4. **Regularly rotate service account keys**
5. **Use OAuth for development, service accounts for production**

## Advanced Configuration

### Custom Authentication Paths

```python
project = Project.create(
    name="my_project",
    backend="gdrive",
    gdrive_folder_id="folder_id",
    gdrive_credentials_path="/custom/path/to/credentials.json",
    gdrive_token_path="/custom/path/to/token.json"
)
```

### Multiple Projects

You can have multiple projects in the same Google Drive folder:

```python
project1 = Project.create(name="project1", backend="gdrive", ...)
project2 = Project.create(name="project2", backend="gdrive", ...)
```

Each will create its own subfolder structure.
