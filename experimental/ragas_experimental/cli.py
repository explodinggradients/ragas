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
from colorama import Fore, Style, init
from ragas_experimental.metric import MetricResult
from .project.core import Project
from .model.pydantic_model import ExtendedPydanticBaseModel as BaseModel

# Initialize colorama
init(autoreset=True)

app = typer.Typer(help="Ragas CLI for running LLM evaluations")

# Create a callback for the main app to make it a group
@app.callback()
def main():
    """Ragas CLI for running LLM evaluations"""
    pass


# Color utility functions
def success(text: str) -> str:
    """Return text in green color for success messages."""
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

def error(text: str) -> str:
    """Return text in red color for error messages."""
    return f"{Fore.RED}{text}{Style.RESET_ALL}"

def info(text: str) -> str:
    """Return text in yellow color for info messages."""
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"

def warning(text: str) -> str:
    """Return text in yellow color for warning messages."""
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"

def metric_pass(text: str) -> str:
    """Return text in green for passing metrics."""
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

def metric_fail(text: str) -> str:
    """Return text in red for failing metrics."""
    return f"{Fore.RED}{text}{Style.RESET_ALL}"

def improvement(text: str) -> str:
    """Return text in green for improvements."""
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

def regression(text: str) -> str:
    """Return text in red for regressions."""
    return f"{Fore.RED}{text}{Style.RESET_ALL}"

def metric_name_color(text: str) -> str:
    """Return text in yellow for metric names."""
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"


def load_eval_module(eval_path: str) -> Any:
    """Load an evaluation module from a file path."""
    eval_path = Path(eval_path).resolve()
    if not eval_path.exists():
        typer.echo(error(f"Error: Evaluation file not found: {eval_path}"))
        raise typer.Exit(1)
    
    # Add the eval directory to Python path so imports work
    eval_dir = eval_path.parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    
    # Load the module
    spec = importlib.util.spec_from_file_location("eval_module", eval_path)
    if spec is None or spec.loader is None:
        typer.echo(error(f"Error: Could not load evaluation file: {eval_path}"))
        raise typer.Exit(1)
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_experiments(project, experiment_func, dataset_name: str, input_data_class: type, baseline_name: Optional[str] = None, metrics: str = None):
    """Run experiments using ragas dataset system."""
    typer.echo(f"Getting dataset: {dataset_name}")
    
    # Get the dataset using project's get_dataset method
    try:
        dataset = project.get_dataset(dataset_name=dataset_name, model=input_data_class)
        dataset.load()  # Load the dataset data
        typer.echo(success(f"✓ Loaded dataset with {len(dataset)} rows"))
    except Exception as e:
        typer.echo(error(f"Error loading dataset '{dataset_name}': {e}"))
        raise typer.Exit(1)
    
    # Run the experiment using the run_async method
    try:
        experiment_result = await experiment_func.run_async(dataset)
        typer.echo(success("✓ Completed experiments successfully"))
    except Exception as e:
        typer.echo(error(f"Error running experiments: {e}"))
        raise typer.Exit(1)
        
    # Handle baseline comparison if specified
    if baseline_name:
        typer.echo(f"Comparing against baseline: {baseline_name}")
        try:
            # The experiment model should be the return type or we can infer it
            baseline = project.get_experiment(baseline_name, model=experiment_result.model)
            # Compare results
            baseline.load()
            
            # Create comparison table
            typer.echo(info("────────────────────────────────────────────────────────────"))
            typer.echo(info(f"dataset   :  {dataset_name}   ({len(dataset)} rows)"))
            typer.echo(info("────────────────────────────────────────────────────────────"))
            
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
            
            # Print numeric metrics table
            failures = 0
            if numeric_metrics:
                typer.echo(info("metric                 current   baseline   Δ         gate"))
                typer.echo(info("───────────────────────────────────────────────────────────"))
                
                for metric_name, values in numeric_metrics.items():
                    current_value = values['current']
                    baseline_value = values['baseline']
                    delta = current_value - baseline_value
                    
                    # Format values
                    current_str = f"{current_value:.3f}".ljust(9)
                    baseline_str = f"{baseline_value:.3f}".ljust(9)
                    
                    # Determine if delta is improvement (depends on metric)
                    is_improvement = delta > 0
                    if "error" in metric_name or "rate" in metric_name:
                        is_improvement = delta < 0
                    
                    # Format delta with arrow and color
                    arrow = "▲" if delta > 0 else "▼"
                    delta_value = f"{arrow}{abs(delta):.3f}"
                    # Pad the original value, then apply color
                    padded_delta = delta_value.ljust(9)
                    if is_improvement:
                        delta_str = improvement(padded_delta)
                    else:
                        delta_str = regression(padded_delta)
                    
                    # Determine if test passes (allow small regression)
                    passed = is_improvement or abs(delta) < 0.01
                    gate_str = metric_pass("pass") if passed else metric_fail("fail")
                    
                    if not passed:
                        failures += 1
                    
                    # Print row
                    metric_display_name = metric_name.replace("_", " ").ljust(20)
                    metric_display_colored = metric_name_color(metric_display_name)
                    typer.echo(f"{metric_display_colored} {current_str} {baseline_str} {delta_str} {gate_str}")
                
                typer.echo(info("────────────────────────────────────────────────────────────"))
            
            # Print categorical metrics
            categorical_failures = 0
            for metric_name_key, values in categorical_metrics.items():
                current_value = values['current'] 
                baseline_value = values['baseline']
                
                typer.echo(f"\n{metric_name_color(metric_name_key)}")
                
                # Get all unique categories
                all_categories = set(current_value.keys()) | set(baseline_value.keys())
                
                for category in sorted(all_categories):
                    current_count = current_value.get(category, 0)
                    baseline_count = baseline_value.get(category, 0)
                    delta = current_count - baseline_count
                    
                    if delta > 0:
                        delta_str = improvement(f"▲{delta}")
                    elif delta < 0:
                        delta_str = regression(f"▼{abs(delta)}")
                    else:
                        delta_str = "→"
                    
                    typer.echo(f"  {category:<15} current: {current_count:<3} baseline: {baseline_count:<3}   {delta_str}")
            
            if categorical_metrics:
                typer.echo(info("────────────────────────────────────────────────────────────"))
            
            
            typer.echo(success("✓ Comparison completed"))
            
        except Exception as e:
            typer.echo(error(f"Error comparing with baseline: {e}"))
            traceback.print_exc()  # Print the full traceback with line numbers
            # Continue without comparison
    else:
        # No baseline provided, just print the current experiment metrics
        typer.echo(info("────────────────────────────────────────────────────────────"))
        typer.echo(info(f"dataset   :  {dataset_name}   ({len(dataset)} rows)"))
        typer.echo(info("────────────────────────────────────────────────────────────"))
        
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
        
        # Print metrics table
        typer.echo(info(f"metric                {experiment_result.name}(current)  "))
        typer.echo(info("─────────────────────────────────"))
        
        for metric_name, metric in agg_metrics.items():
            metric_value = metric.get("score", 0)
            metric_display_name = metric_name.replace("_", " ").ljust(20)
            metric_display = metric_name_color(metric_display_name)
            
            # Handle different metric types for display
            if isinstance(metric_value, dict):
                # Categorical metric - show all values with counts
                if metric_value:
                    # Sort by count (descending) for better readability
                    sorted_items = sorted(metric_value.items(), key=lambda x: x[1], reverse=True)
                    value_parts = [f"{val}({count})" for val, count in sorted_items]
                    value_str = ", ".join(value_parts)
                else:
                    value_str = "N/A"
                typer.echo(f"{metric_display} {value_str}")
            else:
                # Numeric metric
                value_str = f"{metric_value:.3f}".ljust(8)
                typer.echo(f"{metric_display} {value_str}")
            
        typer.echo(info("────────────────────────────────────"))
        typer.echo(success("✓ Experiment results displayed"))

    
    


@app.command()
def evals(
    eval_file: str = typer.Argument(..., help="Path to the evaluation file"),
    dataset: str = typer.Option(..., "--dataset", help="Name of the dataset in the project"),
    metrics: str = typer.Option(..., "--metrics", help="Comma-separated list of metric field names to evaluate"),
    baseline: Optional[str] = typer.Option(None, "--baseline", help="Baseline experiment name to compare against"),
):
    """Run evaluations on a dataset."""
    typer.echo(f"Running evaluation: {eval_file}")
    typer.echo(f"Dataset: {dataset}")
    if baseline:
        typer.echo(f"Baseline: {baseline}")
    
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
            typer.echo(error("Error: No Project instance found in evaluation file"))
            raise typer.Exit(1)
        
        if experiment_func is None:
            typer.echo(error("Error: No experiment function with run_async method found in evaluation file"))
            raise typer.Exit(1)
            
        if input_data_class is None:
            typer.echo(error("Error: Could not determine input data class from experiment function"))
            raise typer.Exit(1)
        
        # Run the experiments
        asyncio.run(run_experiments(project, experiment_func, dataset, input_data_class, baseline, metrics))
        typer.echo(success("✓ Evaluation completed successfully"))
        
    except Exception as e:
        typer.echo(error(f"Error running evaluation: {e}"))
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()