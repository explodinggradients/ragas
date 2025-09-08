"""
Ragas CLI for running experiments from command line.
"""

import asyncio
import importlib.util
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

# from ragas.experimental.project.core import Project  # TODO: Project module not implemented yet
from ragas.utils import console

app = typer.Typer(help="Ragas CLI for running LLM evaluations")


# Create a callback for the main app to make it a group
@app.callback()
def main():
    """Ragas CLI for running LLM evaluations"""
    pass


# Rich utility functions
def success(text: str) -> None:
    """Print text in green color for success messages."""
    console.print(text, style="green")


def error(text: str) -> None:
    """Print text in red color for error messages."""
    console.print(text, style="red")


def info(text: str) -> None:
    """Print text in cyan color for info messages."""
    console.print(text, style="cyan")


def warning(text: str) -> None:
    """Print text in yellow color for warning messages."""
    console.print(text, style="yellow")


def create_numerical_metrics_table(
    metrics_data: Dict[str, Dict], has_baseline: bool = False
) -> Table:
    """Create a Rich table for numerical metrics."""
    table = Table(title="Numerical Metrics")

    # Add columns based on whether we have baseline comparison
    table.add_column("Metric", style="yellow", no_wrap=True)
    table.add_column("Current", justify="right")

    if has_baseline:
        table.add_column("Baseline", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Gate", justify="center")

    for metric_name, values in metrics_data.items():
        current_value = values["current"]

        if has_baseline:
            baseline_value = values["baseline"]
            delta = current_value - baseline_value

            is_improvement = delta > 0
            # Format delta with arrow and color
            arrow = "▲" if delta > 0 else "▼"
            delta_str = f"{arrow}{abs(delta):.3f}"
            delta_color = "green" if is_improvement else "red"

            # Determine if test passes (allow small regression)
            passed = is_improvement or abs(delta) < 0.01
            gate_str = (
                Text("pass", style="green") if passed else Text("fail", style="red")
            )

            table.add_row(
                metric_name.replace("_", " "),
                f"{current_value:.3f}",
                f"{baseline_value:.3f}",
                Text(delta_str, style=delta_color),
                gate_str,
            )
        else:
            table.add_row(metric_name.replace("_", " "), f"{current_value:.3f}")

    return table


def create_categorical_metrics_table(
    metrics_data: Dict[str, Dict], has_baseline: bool = False
) -> Table:
    """Create a Rich table for categorical metrics."""
    table = Table(title="Categorical Metrics")

    # Add columns
    table.add_column("Metric", style="yellow", no_wrap=True)
    table.add_column("Category", style="cyan")
    table.add_column("Current", justify="right")

    if has_baseline:
        table.add_column("Baseline", justify="right")
        table.add_column("Delta", justify="right")

    for metric_name, values in metrics_data.items():
        current_value = values["current"]

        if has_baseline:
            baseline_value = values["baseline"]

            # Get all unique categories
            all_categories = set(current_value.keys()) | set(baseline_value.keys())

            for i, category in enumerate(sorted(all_categories)):
                current_count = current_value.get(category, 0)
                baseline_count = baseline_value.get(category, 0)
                delta = current_count - baseline_count

                if delta > 0:
                    delta_str = Text(f"▲{delta}", style="green")
                elif delta < 0:
                    delta_str = Text(f"▼{abs(delta)}", style="red")
                else:
                    delta_str = Text("→", style="dim")

                # Only show metric name on first row for this metric
                metric_display = metric_name.replace("_", " ") if i == 0 else ""

                table.add_row(
                    metric_display,
                    category,
                    str(current_count),
                    str(baseline_count),
                    delta_str,
                )
        else:
            # Sort by count (descending) for better readability
            if current_value:
                sorted_items = sorted(
                    current_value.items(), key=lambda x: x[1], reverse=True
                )
                for i, (category, count) in enumerate(sorted_items):
                    # Only show metric name on first row for this metric
                    metric_display = metric_name.replace("_", " ") if i == 0 else ""
                    table.add_row(metric_display, category, str(count))
            else:
                table.add_row(metric_name.replace("_", " "), "N/A", "0")

    return table


def extract_metrics_from_experiment(experiment, metric_fields: list) -> Dict[str, list]:
    """Extract metric values from experiment entries."""
    metrics_data = {field_name: [] for field_name in metric_fields}
    for entry in experiment:
        for field_name in metric_fields:
            field_value = getattr(entry, field_name)
            metrics_data[field_name].append(field_value)
    return metrics_data


def calculate_aggregated_metrics(metrics_data: Dict[str, list]) -> Dict[str, Dict]:
    """Calculate aggregated scores for metrics (numeric average or categorical frequency)."""
    agg_metrics = {}
    for metric_name, scores in metrics_data.items():
        # Remove None values
        scores = [score for score in scores if score is not None]
        if not scores:
            avg_score = 0
        elif isinstance(scores[0], (int, float)):
            # Numeric metric - calculate average
            avg_score = sum(scores) / len(scores)
        else:
            # Categorical metric - create frequency distribution
            avg_score = dict(Counter(scores))
        agg_metrics[metric_name] = {"score": avg_score}
    return agg_metrics


def separate_metrics_by_type(
    current_metrics: Dict, baseline_metrics: Optional[Dict] = None
) -> tuple:
    """Separate metrics into numeric and categorical dictionaries."""
    numeric_metrics = {}
    categorical_metrics = {}

    for metric_name, current_metric in current_metrics.items():
        current_value = current_metric.get("score", 0)

        if baseline_metrics and metric_name in baseline_metrics:
            baseline_value = baseline_metrics[metric_name].get("score", 0)

            if isinstance(current_value, dict) and isinstance(baseline_value, dict):
                categorical_metrics[metric_name] = {
                    "current": current_value,
                    "baseline": baseline_value,
                }
            else:
                numeric_metrics[metric_name] = {
                    "current": current_value,
                    "baseline": baseline_value,
                }
        else:
            # No baseline comparison
            if isinstance(current_value, dict):
                categorical_metrics[metric_name] = {"current": current_value}
            else:
                numeric_metrics[metric_name] = {"current": current_value}

    return numeric_metrics, categorical_metrics


def display_metrics_tables(
    numeric_metrics: Dict, categorical_metrics: Dict, has_baseline: bool = False
) -> None:
    """Display metrics tables for numeric and categorical data."""
    if numeric_metrics:
        table = create_numerical_metrics_table(
            numeric_metrics, has_baseline=has_baseline
        )
        console.print(table)

    if categorical_metrics:
        table = create_categorical_metrics_table(
            categorical_metrics, has_baseline=has_baseline
        )
        console.print(table)


def load_eval_module(eval_path: str) -> Any:
    """Load an evaluation module from a file path."""
    eval_path_obj = Path(eval_path).resolve()
    if not eval_path_obj.exists():
        error(f"Error: Evaluation file not found: {eval_path_obj}")
        raise typer.Exit(1)

    # Add the eval directory to Python path so imports work
    eval_dir = eval_path_obj.parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    # Load the module
    spec = importlib.util.spec_from_file_location("eval_module", eval_path_obj)
    if spec is None or spec.loader is None:
        error(f"Error: Could not load evaluation file: {eval_path_obj}")
        raise typer.Exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_experiments(
    project,
    experiment_func,
    dataset_name: str,
    input_data_class: type,
    baseline_name: Optional[str] = None,
    metrics: Optional[str] = None,
    name: Optional[str] = None,
):
    """Run experiments using ragas dataset system."""
    console.print(f"Getting dataset: {dataset_name}")

    # Get the dataset using project's get_dataset method
    try:
        dataset = project.get_dataset(dataset_name=dataset_name, model=input_data_class)
        dataset.load()  # Load the dataset data
        success(f"✓ Loaded dataset with {len(dataset)} rows")
    except Exception as e:
        error(f"Error loading dataset '{dataset_name}': {e}")
        raise typer.Exit(1)

    # Run the experiment using the run_async method
    try:
        experiment_result = await experiment_func.run_async(dataset, name=name)
        success("✓ Completed experiments successfully")
    except Exception as e:
        error(f"Error running experiments: {e}")
        raise typer.Exit(1)

    # Parse metrics from provided list
    metric_fields = [
        metric.strip() for metric in (metrics or "").split(",") if metric.strip()
    ]

    # Extract metrics from current experiment
    current_metrics_data = extract_metrics_from_experiment(
        experiment_result, metric_fields
    )
    current_agg_metrics = calculate_aggregated_metrics(current_metrics_data)

    # Handle baseline comparison if specified
    if baseline_name:
        console.print(f"Comparing against baseline: {baseline_name}")
        try:
            # The experiment model should be the return type or we can infer it
            baseline = project.get_experiment(
                baseline_name, model=experiment_result.model
            )
            baseline.load()

            # Create comparison header with panel
            header_content = f"Experiment: {experiment_result.name}\nDataset: {dataset_name} ({len(dataset)} rows)\nBaseline: {baseline_name}"
            console.print(
                Panel(
                    header_content,
                    title="Ragas Evaluation Results",
                    style="bold white",
                    width=80,
                )
            )

            # Extract metrics from baseline experiment
            baseline_metrics_data = extract_metrics_from_experiment(
                baseline, metric_fields
            )
            baseline_agg_metrics = calculate_aggregated_metrics(baseline_metrics_data)

            # Separate metrics by type with baseline comparison
            numeric_metrics, categorical_metrics = separate_metrics_by_type(
                current_agg_metrics, baseline_agg_metrics
            )

            # Display metrics tables
            display_metrics_tables(
                numeric_metrics, categorical_metrics, has_baseline=True
            )

            success("✓ Comparison completed")

        except Exception as e:
            error(f"Error comparing with baseline: {e}")
            traceback.print_exc()  # Print the full traceback with line numbers
            # Continue without comparison
    else:
        # No baseline provided, just print the current experiment metrics
        header_content = f"Experiment: {experiment_result.name}\nDataset: {dataset_name} ({len(dataset)} rows)"
        console.print(
            Panel(
                header_content,
                title="Ragas Evaluation Results",
                style="bold white",
                width=80,
            )
        )

        # Separate metrics by type without baseline comparison
        numeric_metrics, categorical_metrics = separate_metrics_by_type(
            current_agg_metrics
        )

        # Display metrics tables
        display_metrics_tables(numeric_metrics, categorical_metrics, has_baseline=False)

        success("✓ Experiment results displayed")


@app.command()
def evals(
    eval_file: str = typer.Argument(..., help="Path to the evaluation file"),
    dataset: str = typer.Option(
        ..., "--dataset", help="Name of the dataset in the project"
    ),
    metrics: str = typer.Option(
        ..., "--metrics", help="Comma-separated list of metric field names to evaluate"
    ),
    baseline: Optional[str] = typer.Option(
        None, "--baseline", help="Baseline experiment name to compare against"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", help="Name of the experiment run"
    ),
):
    """Run evaluations on a dataset."""
    console.print(f"Running evaluation: {eval_file}")
    console.print(f"Dataset: {dataset}")
    if baseline:
        console.print(f"Baseline: {baseline}")

    try:
        # Load the evaluation module
        eval_module = load_eval_module(eval_file)

        # Find the project and experiment function
        project = None
        experiment_func = None
        input_data_class = None

        # Look for project and experiment in the module
        for attr_name in dir(eval_module):
            attr = getattr(eval_module, attr_name)
            # TODO: Project class not implemented yet
            # if isinstance(attr, Project):
            #     project = attr
            if hasattr(attr, "get_dataset") and hasattr(attr, "get_experiment"):
                project = attr
            elif hasattr(attr, "run_async"):
                experiment_func = attr
                # Get input type from the experiment function's signature
                import inspect

                sig = inspect.signature(attr)
                if sig.parameters:
                    # Get the first parameter's annotation
                    first_param = next(iter(sig.parameters.values()))
                    if (
                        first_param.annotation
                        and first_param.annotation != inspect.Parameter.empty
                    ):
                        input_data_class = first_param.annotation

        if project is None:
            error("Error: No Project instance found in evaluation file")
            raise typer.Exit(1)

        if experiment_func is None:
            error(
                "Error: No experiment function with run_async method found in evaluation file"
            )
            raise typer.Exit(1)

        if input_data_class is None:
            error(
                "Error: Could not determine input data class from experiment function"
            )
            raise typer.Exit(1)

        # Run the experiments
        asyncio.run(
            run_experiments(
                project,
                experiment_func,
                dataset,
                input_data_class,
                baseline,
                metrics,
                name,
            )
        )
        success("✓ Evaluation completed successfully")

    except Exception as e:
        error(f"Error running evaluation: {e}")
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def hello_world(
    directory: str = typer.Argument(
        ".", help="Directory to run the hello world example in"
    ),
):
    import os
    import time

    import pandas as pd

    if not os.path.exists(directory):
        console.print(f"Directory {directory} does not exist.", style="red")
        raise typer.Exit(1)

    with Live(
        Spinner("dots", text="Creating hello world example...", style="green"),
        console=console,
    ) as live:
        live.update(Spinner("dots", text="Creating directories...", style="green"))
        Path(directory).joinpath("hello_world").mkdir(parents=True, exist_ok=True)
        os.makedirs(os.path.join(directory, "hello_world", "datasets"), exist_ok=True)
        os.makedirs(
            os.path.join(directory, "hello_world", "experiments"), exist_ok=True
        )
        time.sleep(0.5)  # Brief pause to show spinner

        live.update(Spinner("dots", text="Creating test dataset...", style="green"))
        hello_world_data = [
            {
                "id": 1,
                "query": "What is the capital of France?",
                "expected_output": "Paris",
            },
            {"id": 2, "query": "What is 2 + 2?", "expected_output": "4"},
            {
                "id": 3,
                "query": "What is the largest mammal?",
                "expected_output": "Blue Whale",
            },
            {
                "id": 4,
                "query": "Who developed the theory of relativity?",
                "expected_output": "Einstein",
            },
            {
                "id": 5,
                "query": "What is the programming language used for data science?",
                "expected_output": "Python",
            },
            {
                "id": 6,
                "query": "What is the highest mountain in the world?",
                "expected_output": "Mount Everest",
            },
            {
                "id": 7,
                "query": "Who wrote 'Romeo and Juliet'?",
                "expected_output": "Shakespeare",
            },
            {
                "id": 8,
                "query": "What is the fourth planet from the Sun?",
                "expected_output": "Mars",
            },
            {
                "id": 9,
                "query": "What is the name of the fruit that keeps the doctor away?",
                "expected_output": "Apple",
            },
            {
                "id": 10,
                "query": "Who painted the Mona Lisa?",
                "expected_output": "Leonardo da Vinci",
            },
        ]
        df = pd.DataFrame(hello_world_data)
        df.to_csv(
            os.path.join(directory, "hello_world", "datasets", "test_data.csv"),
            index=False,
        )
        time.sleep(0.5)  # Brief pause to show spinner

        live.update(
            Spinner("dots", text="Creating evaluation script...", style="green")
        )
        # Create evals.py file
        evals_content = '''import typing as t

import numpy as np
from pydantic import BaseModel
# from ragas.experimental.project.backends import LocalCSVProjectBackend  # TODO: Not implemented yet
from ragas.metrics.result import MetricResult
from ragas.metrics.numeric import numeric_metric

# TODO: Project class not implemented yet  
# p = Project(
#     project_id="hello_world", 
#     project_backend=LocalCSVProjectBackend("."),
# )


@numeric_metric(name="accuracy_score", allowed_values=(0, 1))
def accuracy_score(response: str, expected: str):
    """
    Is the response a good response to the query?
    """
    result = 1 if expected.lower().strip() == response.lower().strip() else 0
    return MetricResult(
        result=result,
        reason=(
            f"Response contains {expected}"
            if result
            else f"Response does not contain {expected}"
        ),
    )


def mock_app_endpoint(**kwargs) -> str:
    """Mock AI endpoint for testing purposes."""
    mock_responses = [
        "Paris","4","Blue Whale","Einstein","Python","Mount Everest","Shakespeare",
        "Mars","Apple","Leonardo da Vinci",]
    return np.random.choice(mock_responses)


class TestDataRow(BaseModel):
    id: t.Optional[int]
    query: str
    expected_output: str


class ExperimentDataRow(TestDataRow):
    response: str
    accuracy: int
    accuracy_reason: t.Optional[str] = None


# @p.experiment(ExperimentDataRow)  # TODO: Project not implemented
async def run_experiment(row: TestDataRow):
    response = mock_app_endpoint(query=row.query)
    accuracy = accuracy_score.score(response=response, expected=row.expected_output)

    experiment_view = ExperimentDataRow(
        **row.model_dump(),
        response=response,
        accuracy=accuracy.result,
        accuracy_reason=accuracy.reason,
    )
    return experiment_view
'''

        evals_path = os.path.join(directory, "hello_world", "evals.py")
        with open(evals_path, "w") as f:
            f.write(evals_content)
        time.sleep(0.5)  # Brief pause to show spinner

        live.update(Spinner("dots", text="Finalizing hello world example..."))
        time.sleep(0.5)  # Brief pause to show spinner

    hello_world_path = os.path.join(directory, "hello_world")
    success(f"✓ Created hello world example in {hello_world_path}")
    success(
        "✓ You can now run: ragas evals hello_world/evals.py --dataset test_data --metrics accuracy"
    )


if __name__ == "__main__":
    app()
