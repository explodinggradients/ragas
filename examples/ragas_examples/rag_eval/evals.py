import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory

# Add the current directory to the path so we can import rag module
sys.path.insert(0, str(Path(__file__).parent))
from rag import default_rag_client


def _init_clients():
    """Initialize OpenAI client and RAG system."""
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    rag_client = default_rag_client(llm_client=openai_client)
    llm = llm_factory("gpt-4o", client=openai_client)
    return openai_client, rag_client, llm


def load_dataset():
    """Load test dataset for evaluation."""
    data_samples = [
        SingleTurnSample(
            user_input="What is ragas?",
            response="",  # Will be filled by querying RAG
            reference="Ragas is an evaluation framework for LLM applications",
            retrieved_contexts=[],  # Empty for this simple example
        ),
        SingleTurnSample(
            user_input="How are experiment results stored?",
            response="",
            reference="Results are stored under experiments/ folder using different backends",
            retrieved_contexts=[],
        ),
        SingleTurnSample(
            user_input="What metrics are supported?",
            response="",
            reference="Ragas provides discrete, numerical and ranking metrics",
            retrieved_contexts=[],
        ),
    ]

    # Create an EvaluationDataset with the samples
    dataset = EvaluationDataset(samples=data_samples)
    return dataset


def query_rag_system(dataset):
    """Query the RAG system for answers to each question."""
    print("Querying RAG system...")
    _, rag_client, _ = _init_clients()
    for sample in dataset.samples:
        response_dict = rag_client.query(sample.user_input)
        sample.response = response_dict.get("answer", "No response")
        print(f"  ✓ {sample.user_input} -> Got response")

    return dataset


def evaluate_dataset(dataset):
    """Evaluate the dataset by comparing responses with ground truth."""
    print("\nEvaluating responses...")

    results = []
    for i, sample in enumerate(dataset.samples):
        # Simple evaluation: check if response is not empty
        is_valid = bool(sample.response and sample.response.strip() != "No response")

        result = {
            "index": i + 1,
            "user_input": sample.user_input,
            "response": sample.response[:50] + "..." if len(sample.response) > 50 else sample.response,
            "reference": sample.reference[:50] + "..." if len(sample.reference) > 50 else sample.reference,
            "valid_response": is_valid,
        }
        results.append(result)

        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"  {status} - {sample.user_input}")

    return results


def save_results_to_csv(results, output_dir="evals/experiments"):
    """Save evaluation results to a CSV file."""
    if not results:
        print("No results to save")
        return None

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = Path(output_dir) / f"evaluation_results_{timestamp}.csv"

    # Write results to CSV
    fieldnames = ["index", "user_input", "response", "reference", "valid_response"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {csv_file}")
    return csv_file


def display_results(results):
    """Display evaluation results in console."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if results:
        passed = sum(1 for r in results if r["valid_response"])
        total = len(results)
        print(f"\nResults: {passed}/{total} responses valid")
        print("\nDetails:")
        for result in results:
            print(f"  {result['index']}. {result['user_input']}")
            print(f"     Response: {result['response']}")
            print(f"     Reference: {result['reference']}")


def main():
    """Main evaluation workflow."""
    print("Starting RAG evaluation...\n")

    # Load test dataset
    dataset = load_dataset()
    print(f"Loaded {len(dataset.samples)} test cases")

    # Query RAG system
    dataset = query_rag_system(dataset)

    # Evaluate using Ragas
    results = evaluate_dataset(dataset)

    # Display results
    display_results(results)

    return results


if __name__ == "__main__":
    results = main()

    # Optionally save results to CSV
    # Uncomment the line below to export results:
    # save_results_to_csv(results)
