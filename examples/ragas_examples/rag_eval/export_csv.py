import csv
import sys
from datetime import datetime
from pathlib import Path

from evals import load_dataset, query_rag_system, evaluate_dataset


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


def main():
    """Run evaluation and export results to CSV."""
    print("Running evaluation and exporting to CSV...\n")

    # Load test dataset
    dataset = load_dataset()
    print(f"Loaded {len(dataset.samples)} test cases")

    # Query RAG system
    dataset = query_rag_system(dataset)

    # Evaluate using Ragas
    results = evaluate_dataset(dataset)

    # Display results
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

    # Save to CSV
    csv_file = save_results_to_csv(results)

    if csv_file:
        print(f"\n✓ Results exported successfully to CSV")
    else:
        print("\n✗ Failed to export results")
        sys.exit(1)


if __name__ == "__main__":
    main()
