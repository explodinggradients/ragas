"""Example showing how to append data to an existing Google Drive dataset.

This demonstrates the proper pattern for adding data to existing datasets
while preserving the existing records.
"""

from pydantic import BaseModel
from ragas_experimental.dataset import Dataset


# Example data model
class EvaluationRecord(BaseModel):
    question: str
    answer: str
    context: str
    score: float
    feedback: str


def append_to_existing_dataset():
    """Example of appending to an existing dataset."""
    
    folder_id = "folder_id_here"  # Replace with your actual Google Drive folder ID
    
    # Option 1: Load existing dataset and add more data
    print("=== Appending to Existing Dataset ===")
    
    try:
        # Try to load existing dataset
        dataset = Dataset.load(
            name="evaluation_results",
            backend="gdrive",
            data_model=EvaluationRecord,
            folder_id=folder_id,
            credentials_path="credentials.json",
            token_path="token.json"
        )
        print(f"Loaded existing dataset with {len(dataset)} records")
        
    except FileNotFoundError:
        # Dataset doesn't exist, create a new one
        print("Dataset doesn't exist, creating new one")
        dataset = Dataset(
            name="evaluation_results",
            backend="gdrive",
            data_model=EvaluationRecord,
            folder_id=folder_id,
            credentials_path="credentials.json",
            token_path="token.json"
        )
    
    # Show existing records
    print("Existing records:")
    for i, record in enumerate(dataset):
        print(f"  {i+1}. {record.question}")
    
    # Add new records
    new_records = [
        EvaluationRecord(
            question="What is the largest planet in our solar system?",
            answer="Jupiter",
            context="Solar system knowledge question.",
            score=0.9,
            feedback="Correct answer"
        ),
        EvaluationRecord(
            question="Who painted the Mona Lisa?",
            answer="Leonardo da Vinci",
            context="Art history question.",
            score=1.0,
            feedback="Perfect answer"
        )
    ]
    
    # Append new records
    for record in new_records:
        dataset.append(record)
    
    print(f"\nAdded {len(new_records)} new records")
    
    # Save the updated dataset (this replaces the sheet with all records)
    dataset.save()
    print(f"Saved updated dataset with {len(dataset)} total records")
    
    # Verify by listing all records
    print("\nAll records in dataset:")
    for i, record in enumerate(dataset):
        print(f"  {i+1}. {record.question} -> {record.answer}")
    
    return dataset


def create_multiple_datasets():
    """Example of creating separate datasets instead of appending."""
    
    folder_id = "folder_id_here"  # Replace with your actual Google Drive folder ID
    
    print("\n=== Creating Multiple Datasets ===")
    
    # Create different datasets for different evaluation runs
    datasets = {}
    
    for run_name, data in [
        ("basic_qa", [
            EvaluationRecord(
                question="What is 1+1?",
                answer="Two",
                context="Basic math",
                score=1.0,
                feedback="Correct"
            )
        ]),
        ("advanced_qa", [
            EvaluationRecord(
                question="Explain quantum entanglement",
                answer="Quantum entanglement is a phenomenon...",
                context="Advanced physics",
                score=0.8,
                feedback="Good explanation"
            )
        ])
    ]:
        dataset = Dataset(
            name=f"evaluation_{run_name}",
            backend="gdrive",
            data_model=EvaluationRecord,
            folder_id=folder_id,
            credentials_path="credentials.json",
            token_path="token.json"
        )
        
        for record in data:
            dataset.append(record)
        
        dataset.save()
        datasets[run_name] = dataset
        print(f"Created dataset '{run_name}' with {len(dataset)} records")
    
    # List all datasets
    available_datasets = list(datasets.values())[0].backend.list_datasets()
    print(f"\nAll available datasets: {available_datasets}")
    
    return datasets


if __name__ == "__main__":
    try:
        # Method 1: Append to existing dataset
        dataset = append_to_existing_dataset()
        
        # Method 2: Create separate datasets  
        datasets = create_multiple_datasets()
        
        print("\n✅ Append operations completed successfully!")
        print("\nKey points:")
        print("- dataset.save() replaces the entire sheet (this is the intended behavior)")
        print("- To append: load existing data, add new records, then save")
        print("- For different evaluation runs, consider separate datasets")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
