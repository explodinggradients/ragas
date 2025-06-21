"""
Ragas CLI for running experiments from command line.
"""
import asyncio
import importlib.util
import sys
from pathlib import Path
import typer
from typing import Optional, Any, Dict
import traceback
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from ragas_experimental.metric import MetricResult
from .project.core import Project
from .model.pydantic_model import ExtendedPydanticBaseModel as BaseModel
from .utils import console



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


def create_numerical_metrics_table(metrics_data: Dict[str, Dict], has_baseline: bool = False) -> Table:
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
        current_value = values['current']
        
        if has_baseline:
            baseline_value = values['baseline']
            delta = current_value - baseline_value
            
            # Determine if delta is improvement (depends on metric)
            is_improvement = delta > 0
            if "error" in metric_name.lower() or "rate" in metric_name.lower():
                is_improvement = delta < 0
            
            # Format delta with arrow and color
            arrow = "▲" if delta > 0 else "▼"
            delta_str = f"{arrow}{abs(delta):.3f}"
            delta_color = "green" if is_improvement else "red"
            
            # Determine if test passes (allow small regression)
            passed = is_improvement or abs(delta) < 0.01
            gate_str = Text("pass", style="green") if passed else Text("fail", style="red")
            
            table.add_row(
                metric_name.replace("_", " "),
                f"{current_value:.3f}",
                f"{baseline_value:.3f}",
                Text(delta_str, style=delta_color),
                gate_str
            )
        else:
            table.add_row(
                metric_name.replace("_", " "),
                f"{current_value:.3f}"
            )
    
    return table


def create_categorical_metrics_table(metrics_data: Dict[str, Dict], has_baseline: bool = False) -> Table:
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
        current_value = values['current']
        
        if has_baseline:
            baseline_value = values['baseline']
            
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
                    delta_str
                )
        else:
            # Sort by count (descending) for better readability
            if current_value:
                sorted_items = sorted(current_value.items(), key=lambda x: x[1], reverse=True)
                for i, (category, count) in enumerate(sorted_items):
                    # Only show metric name on first row for this metric
                    metric_display = metric_name.replace("_", " ") if i == 0 else ""
                    table.add_row(metric_display, category, str(count))
            else:
                table.add_row(metric_name.replace("_", " "), "N/A", "0")
    
    return table


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


async def run_experiments(project, experiment_func, dataset_name: str, input_data_class: type, baseline_name: Optional[str] = None, metrics: str = None):
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
        experiment_result = await experiment_func.run_async(dataset)
        success("✓ Completed experiments successfully")
    except Exception as e:
        error(f"Error running experiments: {e}")
        raise typer.Exit(1)
        
    # Handle baseline comparison if specified
    if baseline_name:
        console.print(f"Comparing against baseline: {baseline_name}")
        try:
            # The experiment model should be the return type or we can infer it
            baseline = project.get_experiment(baseline_name, model=experiment_result.model)
            # Compare results
            baseline.load()
            
            # Create comparison header with panel
            header_content = f"Experiment: {experiment_result.name}\nDataset: {dataset_name} ({len(dataset)} rows)\nBaseline: {baseline_name}"
            console.print(Panel(header_content, title="Ragas Evaluation Results", style="bold white", width=80))
            
            # Parse metrics from provided list
            current_metric_fields = [metric.strip() for metric in metrics.split(',')]
            current_metrics = {field_name: [] for field_name in current_metric_fields}
            # Iterate through all entries in the current experiment
            for entry in experiment_result:
                for field_name in current_metric_fields:
                    field_value = getattr(entry, field_name)
                    current_metrics[field_name].append(field_value)
            
            # Calculate average scores for each current metric
            current_agg_metrics = {}
            for metric_name in current_metric_fields:
                scores = current_metrics[metric_name]
                if not scores:
                    avg_score = 0
                elif isinstance(scores[0], (int, float)):
                    # Numeric metric - calculate average
                    avg_score = sum(scores) / len(scores)
                else:
                    # Categorical metric - create frequency distribution
                    from collections import Counter
                    avg_score = dict(Counter(scores))
                current_agg_metrics[metric_name] = {"score": avg_score}
            
            # Use same metrics for baseline results
            baseline_metric_fields = current_metric_fields
            baseline_metrics = {field_name: [] for field_name in baseline_metric_fields}
            # Iterate through all entries in the baseline experiment
            for entry in baseline:
                for field_name in baseline_metric_fields:
                    field_value = getattr(entry, field_name)
                    baseline_metrics[field_name].append(field_value)
            
            # Calculate average scores for each baseline metric
            baseline_agg_metrics = {}
            for metric_name in baseline_metric_fields:
                scores = baseline_metrics[metric_name]
                # TODO: remove temporary fix for empty scores
                scores = [score for score in scores if score is not None]
                if not scores:
                    avg_score = 0
                elif isinstance(scores[0], (int, float)):
                    # Numeric metric - calculate average
                    avg_score = sum(scores) / len(scores)
                else:
                    # Categorical metric - create frequency distribution
                    from collections import Counter
                    avg_score = dict(Counter(scores))
                baseline_agg_metrics[metric_name] = {"score": avg_score}
            
            # Separate numeric and categorical metrics
            numeric_metrics = {}
            categorical_metrics = {}
            
            for metric_name, current_metric in current_agg_metrics.items():
                if metric_name in baseline_agg_metrics:
                    current_value = current_metric.get("score", 0)
                    baseline_value = baseline_agg_metrics[metric_name].get("score", 0)
                    
                    if isinstance(current_value, dict) and isinstance(baseline_value, dict):
                        categorical_metrics[metric_name] = {
                            'current': current_value,
                            'baseline': baseline_value
                        }
                    else:
                        numeric_metrics[metric_name] = {
                            'current': current_value,
                            'baseline': baseline_value
                        }
            
            # Display numeric metrics table
            if numeric_metrics:
                table = create_numerical_metrics_table(numeric_metrics, has_baseline=True)
                console.print(table)
            
            # Display categorical metrics table
            if categorical_metrics:
                table = create_categorical_metrics_table(categorical_metrics, has_baseline=True)
                console.print(table)
            
            success("✓ Comparison completed")
            
        except Exception as e:
            error(f"Error comparing with baseline: {e}")
            traceback.print_exc()  # Print the full traceback with line numbers
            # Continue without comparison
    else:
        # No baseline provided, just print the current experiment metrics
        header_content = f"Experiment: {experiment_result.name}\nDataset: {dataset_name} ({len(dataset)} rows)"
        console.print(Panel(header_content, title="Ragas Evaluation Results", style="bold white", width=80))
        
        # Parse metrics from provided list
        metric_fields = [metric.strip() for metric in metrics.split(',')]
        metrics_data = {field_name: [] for field_name in metric_fields}
        # Iterate through all entries in the experiment
        for entry in experiment_result:
            # Get the entry's data as a dict
            for field_name in metric_fields:
                field_value = getattr(entry, field_name)
                metrics_data[field_name].append(field_value)
        
        # Calculate average scores for each metric
        agg_metrics = {}
        for metric_name in metric_fields:
            scores = metrics_data[metric_name]
            # Remove None values like in baseline code
            scores = [score for score in scores if score is not None]
            if not scores:
                avg_score = 0
            elif isinstance(scores[0], (int, float)):
                # Numeric metric - calculate average
                avg_score = sum(scores) / len(scores)
            else:
                # Categorical metric - create frequency distribution
                from collections import Counter
                avg_score = dict(Counter(scores))
            agg_metrics[metric_name] = {"score": avg_score}
        
        # Separate numeric and categorical metrics
        numeric_metrics = {}
        categorical_metrics = {}
        
        for metric_name, metric in agg_metrics.items():
            metric_value = metric.get("score", 0)
            if isinstance(metric_value, dict):
                categorical_metrics[metric_name] = {'current': metric_value}
            else:
                numeric_metrics[metric_name] = {'current': metric_value}
        
        # Display tables
        if numeric_metrics:
            table = create_numerical_metrics_table(numeric_metrics, has_baseline=False)
            console.print(table)
        
        if categorical_metrics:
            table = create_categorical_metrics_table(categorical_metrics, has_baseline=False)
            console.print(table)
            
        success("✓ Experiment results displayed")

    
    


@app.command()
def evals(
    eval_file: str = typer.Argument(..., help="Path to the evaluation file"),
    dataset: str = typer.Option(..., "--dataset", help="Name of the dataset in the project"),
    metrics: str = typer.Option(..., "--metrics", help="Comma-separated list of metric field names to evaluate"),
    baseline: Optional[str] = typer.Option(None, "--baseline", help="Baseline experiment name to compare against"),
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
            if isinstance(attr, Project):
                project = attr
            elif hasattr(attr, 'run_async'):
                experiment_func = attr
                # Get input type from the experiment function's signature
                import inspect
                sig = inspect.signature(attr)
                if sig.parameters:
                    # Get the first parameter's annotation
                    first_param = next(iter(sig.parameters.values()))
                    if first_param.annotation and first_param.annotation != inspect.Parameter.empty:
                        input_data_class = first_param.annotation
        
        if project is None:
            error("Error: No Project instance found in evaluation file")
            raise typer.Exit(1)
        
        if experiment_func is None:
            error("Error: No experiment function with run_async method found in evaluation file")
            raise typer.Exit(1)
            
        if input_data_class is None:
            error("Error: Could not determine input data class from experiment function")
            raise typer.Exit(1)
        
        # Run the experiments
        asyncio.run(run_experiments(project, experiment_func, dataset, input_data_class, baseline, metrics))
        success("✓ Evaluation completed successfully")
        
    except Exception as e:
        error(f"Error running evaluation: {e}")
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()