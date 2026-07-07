"""Load the synthetic memory QA mini-dataset for Ragas evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from ragas import EvaluationDataset

DATASET_PATH = Path(__file__).parent / "datasets" / "memory_qa_synthetic.csv"


def _split_memory_field(value: str) -> List[str]:
    return [chunk.strip() for chunk in value.split("|") if chunk.strip()]


def load_memory_qa_dataset() -> EvaluationDataset:
    rows = []
    with DATASET_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "user_input": row["user_input"],
                    "reference_contexts": _split_memory_field(
                        row["reference_contexts"]
                    ),
                    "retrieved_contexts": _split_memory_field(
                        row["retrieved_contexts"]
                    ),
                    "reference": row["reference"],
                }
            )

    return EvaluationDataset.from_list(rows)


if __name__ == "__main__":
    dataset = load_memory_qa_dataset()
    print(f"Loaded {len(dataset)} synthetic memory QA samples")
    for sample in dataset:
        print(f"- {sample['user_input']}")