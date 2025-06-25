"""
Example usage of the Google Drive backend for Ragas.

This example shows how to:
1. Set up authentication for Google Drive
2. Create a project with Google Drive backend
3. Create and manage datasets stored in Google Sheets

Prerequisites:
1. Install required dependencies:
   pip install google-api-python-client google-auth google-auth-oauthlib

2. Set up Google Drive API credentials:
   - Go to Google Cloud Console
   - Enable Google Drive API and Google Sheets API
   - Create credentials (either OAuth or Service Account)
   - Download the JSON file

3. Set environment variables or provide paths directly
"""

import os
from pydantic import BaseModel
from ragas_experimental.project.core import Project
from ragas_experimental.metric import MetricResult


# Example model for our dataset
class EvaluationEntry(BaseModel):
    question: str
    answer: str
    context: str
    score: float
    feedback: str


def example_oauth_setup():
    """Example using OAuth authentication."""
    
    # Set up environment variables (or pass directly to Project.create)
    # os.environ["GDRIVE_FOLDER_ID"] = "your_google_drive_folder_id_here"
    # os.environ["GDRIVE_CREDENTIALS_PATH"] = "path/to/your/credentials.json"
    
    # Create project with Google Drive backend
    project = Project.create(
        name="my_ragas_project",
        description="A project using Google Drive for storage",
        backend="gdrive",
        gdrive_folder_id="1HLvvtKLnwGWKTely0YDlJ397XPTQ77Yg",
        gdrive_credentials_path="/Users/derekanderson/Downloads/credentials.json",
        gdrive_token_path="token.json"  # Will be created automatically
    )
    
    return project


def example_usage():
    """Example of using the Google Drive backend."""
    
    # Create a project (choose one of the authentication methods above)
    project = example_oauth_setup()  # or example_service_account_setup()
    
    # Create a dataset
    dataset = project.create_dataset(
        model=EvaluationEntry,
        name="evaluation_results"
    )
    
    # Add some entries
    entry1 = EvaluationEntry(
        question="What is the capital of France?",
        answer="Paris",
        context="France is a country in Europe.",
        score=0.95,
        feedback="Correct answer"
    )
    
    entry2 = EvaluationEntry(
        question="What is 2+2?",
        answer="4",
        context="Basic arithmetic question.",
        score=1.0,
        feedback="Perfect answer"
    )
    
    # Append entries to the dataset
    dataset.append(entry1)
    dataset.append(entry2)
    
    # Load all entries
    dataset.load()
    print(f"Dataset contains {len(dataset)} entries")
    
    # Access entries
    for i, entry in enumerate(dataset):
        print(f"Entry {i}: {entry.question} -> {entry.answer} (Score: {entry.score})")
    
    # Update an entry
    dataset[0].score = 0.98
    dataset[0].feedback = "Updated feedback"
    dataset[0] = dataset[0]  # Trigger update
    
    # Search for entries
    entry = dataset._backend.get_entry_by_field("question", "What is 2+2?", EvaluationEntry)
    if entry:
        print(f"Found entry: {entry.answer}")
    
    return dataset


if __name__ == "__main__":
    # Run the example
    try:
        dataset = example_usage()
        print("Google Drive backend example completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Install required dependencies")
        print("2. Set up Google Drive API credentials")
        print("3. Update the folder ID and credential paths in this example")
