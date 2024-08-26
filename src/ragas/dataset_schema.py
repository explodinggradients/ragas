from typing import Dict, List, Optional

from datasets import Dataset
from langchain_core.pydantic_v1 import BaseModel


class EvaluationSample(BaseModel):
    user_input: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    ground_truth_contexts: Optional[List[str]] = None
    response: Optional[str] = None
    ground_truth: Optional[str] = None
    rubric: Optional[Dict] = None

    def to_dict(self):
        row = {
            "user_input": self.user_input,
            "retrieved_contexts": self.retrieved_contexts,
            "ground_truth_contexts": self.ground_truth_contexts,
            "response": self.response,
            "ground_truth": self.ground_truth,
            "rubric": self.rubric,
        }
        row = {k: v for k, v in row.items() if v is not None}
        return row


class EvaluationDataset(BaseModel):
    samples: List[EvaluationSample]

    def to_hf_dataset(self):
        rows = [sample.to_dict() for sample in self.samples]
        return Dataset.from_list(rows)

    def features(self):
        return self.to_hf_dataset().features.keys()
