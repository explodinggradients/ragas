import argparse
import asyncio
import datetime
import json
import os
import sys
from typing import List, Optional

import pandas as pd
from run_prompt import run_prompt

from ragas import Dataset, experiment
from ragas.metrics import MetricResult, discrete_metric


@discrete_metric(name="labels_exact_match", allowed_values=["correct", "incorrect"])
def labels_exact_match(prediction: str, expected_labels: str):
    """Check if the predicted labels exactly match the expected labels."""
    try:
        parsed_json = json.loads(prediction)
        predicted_labels = parsed_json.get("labels", [])

        # Convert to sets for comparison (handle order independence)
        predicted_set = set(predicted_labels)
        expected_set = set(expected_labels.split(";")) if expected_labels else set()

        if predicted_set == expected_set:
            return MetricResult(
                value="correct",
                reason=f"Correctly predicted labels: {sorted(list(predicted_set))}",
            )
        else:
            return MetricResult(
                value="incorrect",
                reason=f"Expected labels: {sorted(list(expected_set))}; Got labels: {sorted(list(predicted_set))}",
            )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return MetricResult(
            value="incorrect",
            reason=f"Failed to parse labels from response: {str(e)}",
        )


@discrete_metric(name="priority_accuracy", allowed_values=["correct", "incorrect"])
def priority_accuracy(prediction: str, expected_priority: str):
    """Check if the predicted priority matches the expected priority."""
    try:
        parsed_json = json.loads(prediction)
        predicted_priority = parsed_json.get("priority")

        if predicted_priority == expected_priority:
            return MetricResult(
                value="correct",
                reason=f"Correctly predicted priority: {expected_priority}",
            )
        else:
            return MetricResult(
                value="incorrect",
                reason=f"Expected priority: {expected_priority}; Got priority: {predicted_priority}",
            )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return MetricResult(
            value="incorrect",
            reason=f"Failed to parse priority from response: {str(e)}",
        )


@experiment()
async def support_triage_experiment(row, prompt_file: str, experiment_name: str):
    """Experiment function for support triage evaluation."""
    # Get model response
    response = await asyncio.to_thread(run_prompt, row["text"], prompt_file=prompt_file)

    # Parse response to extract predicted values
    try:
        parsed_json = json.loads(response)
        predicted_labels = parsed_json.get("labels", [])
        predicted_priority = parsed_json.get("priority")

        # Convert predicted labels back to semicolon-separated string for consistency
        predicted_labels_str = ";".join(predicted_labels) if predicted_labels else ""
    except Exception:
        predicted_labels_str = ""
        predicted_priority = None

    # Score the response
    labels_score = labels_exact_match.score(
        prediction=response, expected_labels=row["labels"]
    )
    priority_score = priority_accuracy.score(
        prediction=response, expected_priority=row["priority"]
    )

    return {
        "id": row["id"],
        "text": row["text"],
        "response": response,
        "experiment_name": experiment_name,
        "expected_labels": row["labels"],
        "predicted_labels": predicted_labels_str,
        "expected_priority": row["priority"],
        "predicted_priority": predicted_priority,
        "labels_score": labels_score.value,
        "priority_score": priority_score.value,
    }


def load_dataset():
    """Load the support triage dataset from CSV file."""
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "datasets", "support_triage.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    # Read CSV and create Dataset
    df = pd.read_csv(dataset_path)

    # Validate required columns
    required_cols = ["id", "text", "labels", "priority"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Create Ragas Dataset
    dataset = Dataset(name="support_triage", backend="local/csv", root_dir=".")

    for _, row in df.iterrows():
        dataset.append(
            {
                "id": str(row["id"]),
                "text": row["text"],
                "labels": row["labels"],
                "priority": row["priority"],
            }
        )

    return dataset


def compare_inputs_to_output(
    inputs: List[str], output_path: Optional[str] = None
) -> str:
    """Compare multiple experiment CSVs and write a combined CSV.

    - Requires 'id' column in all inputs; uses it as the alignment key
    - Builds output with id + canonical columns + per-experiment response/score columns
    - Returns the full output path
    """
    if not inputs or len(inputs) < 2:
        raise ValueError("At least two input CSV files are required for comparison")

    # Load all inputs
    dataframes = []
    experiment_names = []
    for path in inputs:
        df = pd.read_csv(path)
        if "experiment_name" not in df.columns:
            raise ValueError(f"Missing 'experiment_name' column in {path}")
        exp_name = str(df["experiment_name"].iloc[0])
        experiment_names.append(exp_name)
        dataframes.append(df)

    canonical_cols = ["text", "expected_labels", "expected_priority"]
    base_df = dataframes[0]

    # Require 'id' in all inputs
    if not all("id" in df.columns for df in dataframes):
        raise ValueError(
            "All input CSVs must contain an 'id' column to align rows. Re-run experiments after adding 'id' to your dataset."
        )

    # Validate duplicates and matching sets of IDs
    key_sets = []
    for idx, df in enumerate(dataframes):
        keys = df["id"].astype(str)
        if keys.duplicated().any():
            dupes = keys[keys.duplicated()].head(3).tolist()
            raise ValueError(
                f"Input {inputs[idx]} contains duplicate id values. Examples: {dupes}"
            )
        key_sets.append(set(keys.tolist()))

    base_keys = key_sets[0]
    for i, ks in enumerate(key_sets[1:], start=1):
        if ks != base_keys:
            missing_in_other = list(base_keys - ks)[:5]
            missing_in_base = list(ks - base_keys)[:5]
            raise ValueError(
                "Inputs do not contain the same set of IDs.\n"
                f"- Missing in file {i + 1}: {missing_in_other}\n"
                f"- Extra in file {i + 1}: {missing_in_base}"
            )

    # Validate canonical columns exist in base
    missing = [c for c in canonical_cols if c not in base_df.columns]
    if missing:
        raise ValueError(f"First CSV missing required columns: {missing}")

    # Build combined on base order using 'id' as alignment key
    base_ids_str = base_df["id"].astype(str)
    combined = base_df[["id"] + canonical_cols].copy()

    # Append per-experiment outputs by aligned ID
    for df, exp_name in zip(dataframes, experiment_names):
        df = df.copy()
        df["id"] = df["id"].astype(str)
        df = df.set_index("id")
        for col in ["response", "labels_score", "priority_score"]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in one input. Please provide per-row '{col}'."
                )
        combined[f"{exp_name}_response"] = base_ids_str.map(df["response"])
        combined[f"{exp_name}_labels_score"] = base_ids_str.map(df["labels_score"])
        combined[f"{exp_name}_priority_score"] = base_ids_str.map(df["priority_score"])

    # Determine output path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(current_dir, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    if output_path is None or output_path.strip() == "":
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(experiments_dir, f"{run_id}-comparison.csv")
    else:
        # If relative path, place under experiments dir
        if not os.path.isabs(output_path):
            output_path = os.path.join(experiments_dir, output_path)

    # Sort by id for user-friendly reading
    if "id" in combined.columns:
        combined = combined.sort_values(by="id").reset_index(drop=True)
    combined.to_csv(output_path, index=False)

    # Print per-experiment accuracy summary
    for df, exp_name in zip(dataframes, experiment_names):
        try:
            labels_acc = (df["labels_score"] == "correct").mean()
            priority_acc = (df["priority_score"] == "correct").mean()
            print(f"{exp_name} Labels Accuracy: {labels_acc:.2%}")
            print(f"{exp_name} Priority Accuracy: {priority_acc:.2%}")
        except Exception:
            pass

    return output_path


async def run_command(prompt_file: str, name: Optional[str]) -> None:
    """Run a single experiment using the provided prompt file and name."""
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ Error: OpenAI API key not found!")
        print("Please set your API key: export OPENAI_API_KEY=your_actual_key")
        return

    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    prompt_name = os.path.splitext(os.path.basename(prompt_file))[0]
    exp_name = name or prompt_name

    # Ensure output directory exists (experiment framework saves under experiments/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(current_dir, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    print(f"Running evaluation with prompt file: {prompt_file}")
    results = await support_triage_experiment.arun(
        dataset,
        name=f"{run_id}-{exp_name}",
        prompt_file=prompt_file,
        experiment_name=exp_name,
    )
    print(f"✅ {exp_name}: {len(results)} cases evaluated")
    print(f"Results saved to: {os.path.join(experiments_dir, results.name)}.csv")

    # Accuracy summary
    labels_accuracy = sum(1 for r in results if r["labels_score"] == "correct") / max(
        1, len(results)
    )
    priority_accuracy = sum(
        1 for r in results if r["priority_score"] == "correct"
    ) / max(1, len(results))
    print(f"{exp_name} Labels Accuracy: {labels_accuracy:.2%}")
    print(f"{exp_name} Priority Accuracy: {priority_accuracy:.2%}")


def compare_command(inputs: List[str], output: Optional[str]) -> None:
    output_path = compare_inputs_to_output(inputs, output)
    print(f"Combined comparison saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Support Triage Prompt Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument(
        "--prompt_file", type=str, required=True, help="Prompt file to evaluate"
    )
    run_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (defaults to prompt filename)",
    )

    # compare subcommand
    cmp_parser = subparsers.add_parser(
        "compare", help="Combine multiple experiment CSVs"
    )
    cmp_parser.add_argument(
        "--inputs", nargs="+", required=True, help="Input CSV files to compare"
    )
    cmp_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (defaults to experiments/<timestamp>-comparison.csv)",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_command(prompt_file=args.prompt_file, name=args.name))
        sys.exit(0)
    elif args.command == "compare":
        compare_command(inputs=args.inputs, output=args.output)
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(2)
