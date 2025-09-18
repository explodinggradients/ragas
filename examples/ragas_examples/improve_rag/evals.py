"""
Evaluation script for SimpleRAG using HuggingFace documentation Q&A dataset.
This evaluates the SimpleRAG system against a ground truth dataset.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import datasets
import mlflow
from dotenv import load_dotenv
from openai import OpenAI

from ragas import Dataset, experiment
from ragas.llms import instructor_llm_factory
from ragas.metrics import DiscreteMetric

from .simple_rag import create_rag_client

# Load environment variables
load_dotenv(".env")

def download_and_save_dataset() -> Path:
    """
    Download the HuggingFace doc Q&A dataset and save locally.
    
    Returns:
        Path to the saved dataset CSV file
    """
    # Create datasets directory relative to current working directory
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    dataset_path = datasets_dir / "hf_doc_qa_eval.csv"
    
    if dataset_path.exists():
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    print("Downloading HuggingFace doc Q&A evaluation dataset...")
    hf_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
    
    # Convert to pandas and save as CSV
    df = hf_dataset.to_pandas()
    df.to_csv(dataset_path, index=False)
    
    print(f"Dataset saved to {dataset_path} with {len(df)} samples")
    return dataset_path


def create_ragas_dataset(dataset_path: Path) -> Dataset:
    """
    Create a Ragas Dataset from the downloaded CSV file.
    
    Args:
        dataset_path: Path to the CSV file
        
    Returns:
        Ragas Dataset instance
    """
    dataset = Dataset(
        name="hf_doc_qa_eval",
        backend="local/csv",
        root_dir=".",  # Use current directory to avoid nested datasets folder
    )
    
    # Load the CSV data
    import pandas as pd
    df = pd.read_csv(dataset_path)
    
    # Add rows to the dataset
    for _, row in df.iterrows():
        dataset_row = {
            "question": row["question"],
            "context": row["context"],
            "expected_answer": row["expected_answer"],
        }
        dataset.append(dataset_row)
    
    # Save the dataset
    dataset.save()
    print(f"Created Ragas dataset with {len(df)} samples")
    return dataset


# Initialize OpenAI client and models
def get_openai_client():
    """Get OpenAI client with proper error handling."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set your OpenAI API key: export OPENAI_API_KEY='your_key'"
        )
    return OpenAI(api_key=api_key)

openai_client = get_openai_client()
rag_client = create_rag_client(model="gpt-5-mini")
llm = instructor_llm_factory("openai", model="gpt-5-mini", client=openai_client)


# Define correctness metric
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer and determine if it's correct.
    
Consider the response correct if it:
1. Contains the key information from the expected answer
2. Is factually accurate based on the provided context
3. Adequately addresses the question asked

Return 'pass' if the response is correct, 'fail' if it's incorrect.

Question: {question}
Expected Answer: {expected_answer}
Model Response: {response}

Evaluation:""",
    allowed_values=["pass", "fail"],
)


@experiment()
async def evaluate_rag(row: Dict[str, Any], agentic_rag: bool = False) -> Dict[str, Any]:
    """
    Run RAG evaluation on a single row.
    
    Args:
        row: Dictionary containing question, context, and expected_answer
        agentic_rag: If True, use agentic RAG; otherwise use simple RAG
        
    Returns:
        Dictionary with evaluation results
    """
    question = row["question"]
    
    if agentic_rag:
        # Use agentic RAG
        from .agentic_rag import query_agentic_rag
        model_response = await query_agentic_rag(question)
        # For agentic RAG, we don't have retrieved_documents in the same format
        rag_response = {"answer": model_response, "retrieved_documents": []}
    else:
        # Use simple RAG - using asyncio.to_thread for better concurrency
        rag_response = await asyncio.to_thread(rag_client.query, question, top_k=4)
        model_response = rag_response.get("answer", "")
    
    # Evaluate correctness - use asyncio.to_thread to avoid blocking the event loop
    score = await asyncio.to_thread(
        correctness_metric.score,
        question=question,
        expected_answer=row["expected_answer"],
        response=model_response,
        context=row["context"],
        llm=llm
    )
    
    # Return evaluation results
    result = {
        **row,
        "model_response": model_response,
        "correctness_score": score.value,
        "correctness_reason": score.reason,
        "retrieved_documents": [
            doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            for doc in rag_response.get("retrieved_documents", [])
        ]
    }
    
    return result


async def main(test_mode: bool = False, agentic_rag: bool = False):
    """
    Main function to run the complete evaluation pipeline.
    
    Args:
        test_mode: If True, run on only a small subset for testing
        agentic_rag: If True, use agentic RAG; otherwise use simple RAG
    """
    try:
        # Set up experiment name with timestamp and mode
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mode_label = "agenticrag" if agentic_rag else "simplerag"
        exp_name = f"{timestamp}_{mode_label}"
        
        # MLflow is already configured in the individual RAG files
        
        mode_name = "Agentic RAG" if agentic_rag else "Simple RAG"
        print(f"=== {mode_name} Evaluation Pipeline ===")
        
        # Step 1: Download and prepare dataset
        print("\n1. Downloading dataset...")
        dataset_path = download_and_save_dataset()
        
        # Step 2: Create Ragas dataset
        print("\n2. Creating Ragas dataset...")
        dataset = create_ragas_dataset(dataset_path)
        
        # Step 2.5: Initialize BM25 retriever upfront to avoid hanging during evaluation
        print("\n2.5. Initializing BM25 retriever (this may take a moment)...")
        from .data_utils import get_bm25_retriever
        _ = get_bm25_retriever()  # This will cache the retriever
        print("BM25 retriever initialized successfully!")
        
        # Step 3: Prepare dataset for evaluation
        eval_dataset = dataset
        if test_mode:
            print("\n⚠️  Running in TEST MODE - evaluating only first 3 samples")
            # Create a smaller dataset for testing
            test_samples = []
            for i, sample in enumerate(dataset):
                if i >= 3:  # Only take first 3 samples
                    break
                test_samples.append(sample)
            
            # Create new test dataset
            eval_dataset = Dataset(
                name="hf_doc_qa_eval_test",
                backend="local/csv", 
                root_dir="."  # Use current directory to avoid nested datasets folder
            )
            for sample in test_samples:
                eval_dataset.append(sample)
            eval_dataset.save()
        
        # Step 4: Run evaluation experiment
        print(f"\n3. Running {mode_name} evaluation on {len(eval_dataset)} samples...")
        print("This may take several minutes depending on the dataset size...")
        
        experiment_results = await evaluate_rag.arun(
            eval_dataset, 
            name=exp_name,
            agentic_rag=agentic_rag
        )
        
        # Step 4: Print summary
        print("\n=== Evaluation Results ===")
        
        if experiment_results and len(experiment_results) > 0:
            # Calculate pass rate
            pass_count = sum(1 for result in experiment_results if result.get("correctness_score") == "pass")
            total_count = len(experiment_results)
            pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0
            
            print(f"Total samples evaluated: {total_count}")
            print(f"Passed: {pass_count}")
            print(f"Failed: {total_count - pass_count}")
            print(f"Pass rate: {pass_rate:.1f}%")
        else:
            print("No results generated. Please check for errors.")
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    
    # Check for flags
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    agentic_rag = "--agentic-rag" in sys.argv
    
    if test_mode:
        print("🧪 Running in TEST MODE (first 3 samples only)")
    
    if agentic_rag:
        print("🤖 Running with AGENTIC RAG")
    else:
        print("📚 Running with SIMPLE RAG")
    
    asyncio.run(main(test_mode=test_mode, agentic_rag=agentic_rag))
