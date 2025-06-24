__all__ = [
    "create_nano_id",
    "async_to_sync",
    "plot_experiments_as_subplots",
    "get_test_directory",
]

import asyncio
import functools
import os
import string
import tempfile
import uuid
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_nano_id(size=12):
    # Define characters to use (alphanumeric)
    alphabet = string.ascii_letters + string.digits

    # Generate UUID and convert to int
    uuid_int = uuid.uuid4().int

    # Convert to base62
    result = ""
    while uuid_int:
        uuid_int, remainder = divmod(uuid_int, len(alphabet))
        result = alphabet[remainder] + result

    # Pad if necessary and return desired length
    return result[:size]


def async_to_sync(async_func):
    """Convert an async function to a sync function"""

    @functools.wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(async_func(*args, **kwargs))
        except RuntimeError:
            return asyncio.run(async_func(*args, **kwargs))

    return sync_wrapper


def plot_experiments_as_subplots(data, experiment_names=None):
    """
    Plot metrics comparison across experiments.

    Parameters:
    - data: Dictionary with experiment_names as keys and metrics as nested dictionaries
    - experiment_names: List of experiment IDs in the order they should be plotted

    Returns:
    - Plotly figure object with horizontal subplots
    """
    if experiment_names is None:
        experiment_names = list(data.keys())

    exp_short_names = [f"{name[:10]}.." for name in experiment_names]
    # TODO: need better solution to identify what type of metric it is
    # this is a temporary solution
    # Identify metrics and their types
    metrics = {}
    for exp_id in experiment_names:
        for metric_name, values in data[exp_id].items():
            # Classify metric type (discrete or numerical)
            if metric_name not in metrics:
                # Check first value to determine type
                is_discrete = isinstance(values[0], str)
                metrics[metric_name] = {
                    "type": "discrete" if is_discrete else "numerical"
                }

    # Create horizontal subplots (one for each metric)
    fig = make_subplots(
        rows=1,
        cols=len(metrics),
        subplot_titles=[
            f"{metric.capitalize()} Comparison" for metric in metrics.keys()
        ],
        horizontal_spacing=0.1,
    )

    # Process metrics and add traces
    col_idx = 1
    for metric_name, metric_info in metrics.items():
        if metric_info["type"] == "discrete":
            # For discrete metrics (like pass/fail)
            categories = set()
            for exp_id in experiment_names:
                count = Counter(data[exp_id][metric_name])
                categories.update(count.keys())

            categories = sorted(list(categories))

            for category in categories:
                y_values = []
                for exp_id in experiment_names:
                    count = Counter(data[exp_id][metric_name])
                    total = sum(count.values())
                    percentage = (count.get(category, 0) / total) * 100
                    y_values.append(percentage)

                # Assign colors based on category

                # Generate consistent color for other categories
                import hashlib

                hash_obj = hashlib.md5(category.encode())
                hash_hex = hash_obj.hexdigest()
                color = f"#{hash_hex[:6]}"

                fig.add_trace(
                    go.Bar(
                        x=exp_short_names,
                        y=y_values,
                        name=category.capitalize(),
                        marker_color=color,
                        width=0.5,  # Narrower bars
                        hoverinfo="text",
                        hovertext=[
                            f"{category.capitalize()}: {x:.1f}%" for x in y_values
                        ],
                        showlegend=False,  # Remove legend
                    ),
                    row=1,
                    col=col_idx,
                )

        else:  # Numerical metrics
            normalized_values = []
            original_values = []

            for exp_id in experiment_names:
                values = data[exp_id][metric_name]
                mean_val = np.mean(values)
                original_values.append(mean_val)

                # Normalize to 0-100 scale
                min_val = np.min(values)
                max_val = np.max(values)
                normalized = ((mean_val - min_val) / (max_val - min_val)) * 100
                normalized_values.append(normalized)

            # Add bar chart for numerical data
            fig.add_trace(
                go.Bar(
                    x=exp_short_names,
                    y=normalized_values,
                    name=metric_name.capitalize(),
                    marker_color="#2E8B57",  # Sea green
                    width=0.5,  # Narrower bars
                    hoverinfo="text",
                    hovertext=[
                        f"{metric_name.capitalize()} Mean: {val:.2f} (Normalized: {norm:.1f}%)"
                        for val, norm in zip(original_values, normalized_values)
                    ],
                    showlegend=False,  # Remove legend
                ),
                row=1,
                col=col_idx,
            )

        # Update axes for each subplot
        fig.update_yaxes(
            title_text=(
                "Percentage (%)"
                if metric_info["type"] == "discrete"
                else "Normalized Value"
            ),
            range=[0, 105],  # Leave room for labels at the top
            ticksuffix="%",
            showgrid=True,
            gridcolor="lightgray",
            showline=True,
            linewidth=1,
            linecolor="black",
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(
            title_text="Experiments",
            tickangle=-45,
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            row=1,
            col=col_idx,
        )

        col_idx += 1

    # Update layout for the entire figure
    fig.update_layout(
        title="Experiment Comparison by Metrics",
        barmode=(
            "stack"
            if any(
                metric_info["type"] == "discrete" for metric_info in metrics.values()
            )
            else "group"
        ),
        height=400,  # Reduced height
        width=250 * len(metrics) + 150,  # Adjust width based on number of metrics
        showlegend=False,  # Remove legend
        margin=dict(t=80, b=50, l=50, r=50),
        plot_bgcolor="white",
        hovermode="closest",
    )

    return fig


# Helper function for tests
def get_test_directory():
    """Create a test directory that will be cleaned up on process exit.

    Returns:
        str: Path to test directory
    """
    # Create a directory in the system temp directory
    test_dir = os.path.join(tempfile.gettempdir(), f"ragas_test_{create_nano_id()}")
    os.makedirs(test_dir, exist_ok=True)

    return test_dir
