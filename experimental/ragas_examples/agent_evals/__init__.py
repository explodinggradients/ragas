"""Agent Evaluation Example

This module contains a mathematical agent that can solve complex expressions
using atomic operations and function calling capabilities.
"""

from .agent import MathToolsAgent, get_default_agent
from .evals import run_experiment, load_dataset

__all__ = ["MathToolsAgent", "get_default_agent", "run_experiment", "load_dataset"]