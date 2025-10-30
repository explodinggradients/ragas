import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

from evals import load_dataset, run_experiment


def save_results_to_csv(results, output_dir="evals/experiments"):
    """Save experiment results to a CSV file."""
    if not results:
        print("No results to save")
        return None

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = Path(output_dir) / f"evaluation_results_{timestamp}.csv"

    # Extract fieldnames from first result
    if results:
        fieldnames = list(results[0].keys())
    else:
        fieldnames = []

    # Write results to CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {csv_file}")
    return csv_file


async def main():
    """Run evaluation and export results to CSV."""
    print("Running experiment and exporting to CSV...\n")

    # Load test dataset
    dataset = load_dataset()
    print(f"Loaded {len(list(dataset))} test cases")

    # Run experiment with DiscreteMetric
    print("Running experiments with DiscreteMetric evaluation...\n")
    results = await run_experiment.arun(dataset)

    # Convert results to list if needed
    if hasattr(results, '__iter__') and not isinstance(results, list):
        results = list(results)

    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if results:
        print(f"\nTotal experiments: {len(results)}")
        print("\nDetails:")
        for i, result in enumerate(results, 1):
            question = result.get("question", "N/A")
            response = result.get("response", "N/A")
            score = result.get("score", "N/A")
            print(f"  {i}. {question}")
            print(f"     Response: {response[:100]}...")
            print(f"     Score: {score}")

    # Save to CSV
    csv_file = save_results_to_csv(results)

    if csv_file:
        print(f"\n✓ Results exported successfully to CSV")
    else:
        print("\n✗ Failed to export results")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
