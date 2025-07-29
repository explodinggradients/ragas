"""Example usage of the Google Drive backend for Ragas.

This example shows how to:
1. Set up authentication for Google Drive
2. Create a dataset with Google Drive backend
3. Store and retrieve data from Google Sheets

Prerequisites:
1. Install Google Drive dependencies:
   pip install "ragas_experimental[gdrive]"

2. Set up Google Drive API credentials:
   - Go to Google Cloud Console
   - Enable Google Drive API and Google Sheets API
   - Create credentials (OAuth or Service Account)
   - Download the JSON file

3. Set up authentication - choose one:
   Option A: Environment variables
   Option B: Pass paths directly to backend

For detailed setup instructions, see the documentation.
"""

import os
from pydantic import BaseModel

from ragas_experimental.dataset import Dataset


# Example data model
class EvaluationRecord(BaseModel):
    question: str
    answer: str
    context: str
    score: float
    feedback: str


def example_usage():
    """Example of using the Google Drive backend."""
    
    # REQUIRED: Replace with your actual Google Drive folder ID
    # This should be the ID from the Google Drive folder URL:
    # https://drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE
    folder_id = "folder_id_here"
    
    # Option A: Set up with environment variables
    # os.environ["GDRIVE_CREDENTIALS_PATH"] = "path/to/credentials.json"
    # dataset = Dataset(
    #     name="evaluation_results",
    #     backend="gdrive",
    #     data_model=EvaluationRecord,  # This is required when using Pydantic models
    #     folder_id=folder_id
    # )
    
    # Option B: Pass credentials directly
    dataset = Dataset(
        name="evaluation_results",
        backend="gdrive",
        data_model=EvaluationRecord,  # This is required when using Pydantic models
        folder_id=folder_id,
        credentials_path="credentials.json",  # For OAuth
        # service_account_path="path/to/service_account.json",  # Alternative: Service Account
        token_path="token.json"  # Where OAuth token will be saved
    )
    
    # Create some sample data
    sample_data = [
        EvaluationRecord(
            question="What is the capital of France?",
            answer="Paris",
            context="France is a country in Western Europe.",
            score=0.95,
            feedback="Correct answer"
        ),
        EvaluationRecord(
            question="What is 2 + 2?",
            answer="Four",  # Changed from "4" to avoid Google Sheets auto-conversion to number
            context="Basic arithmetic question.",
            score=1.0,
            feedback="Perfect answer"
        ),
        EvaluationRecord(
            question="Who wrote Romeo and Juliet?",
            answer="William Shakespeare",
            context="Romeo and Juliet is a famous play.",
            score=1.0,
            feedback="Correct author"
        )
    ]
    
    # Add data to the dataset
    for record in sample_data:
        dataset.append(record)
    
    # Save to Google Drive
    dataset.save()
    print(f"Saved {len(dataset)} records to Google Drive")
    
    # Load data back
    dataset.reload()
    print(f"Loaded {len(dataset)} records from Google Drive")
    
    # Access individual records
    for i, record in enumerate(dataset):
        print(f"Record {i+1}: {record.question} -> {record.answer} (Score: {record.score})")
    
    # List all datasets in the backend
    available_datasets = dataset.backend.list_datasets()
    print(f"Available datasets: {available_datasets}")
    
    return dataset


if __name__ == "__main__":
    try:
        dataset = example_usage()
        print("\nGoogle Drive backend example completed successfully!")
        print("\nYour data is now stored in Google Sheets within your specified folder.")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Install required dependencies: pip install 'ragas_experimental[gdrive]'")
        print("2. Set up Google Drive API credentials")
        print("3. Update the folder_id and credential paths in this example")
        print("4. Ensure the Google Drive folder is accessible to your credentials")
