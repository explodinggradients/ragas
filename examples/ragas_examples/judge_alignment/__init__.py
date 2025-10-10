"""LLM-as-judge alignment evaluation example.

Functions:
- load_dataset: Load annotated dataset with human judgments
- judge_experiment: Run evaluation (Judge → Compare)
- judge_alignment: Alignment metric comparing judge and human labels

Metrics:
- accuracy_metric: Baseline judge metric
- accuracy_metric_v2: Improved judge metric with few-shot examples
"""

from .evals import (
    load_dataset,
    judge_experiment,
    judge_alignment,
    accuracy_metric,
    accuracy_metric_v2,
)

__all__ = [
    "load_dataset",
    "judge_experiment",
    "judge_alignment",
    "accuracy_metric",
    "accuracy_metric_v2",
]


