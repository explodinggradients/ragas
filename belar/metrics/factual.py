from __future__ import annotations

import typing as t
from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from belar.metrics import Metric
from belar.utils import device_check


@dataclass
class EntailmentScore(Metric):
    """
    Entailment score using ground truth as premise and generated text as hypothesis.
    """

    model_name: str = "typeform/distilbert-base-uncased-mnli"
    max_length: int = 512
    batch_size: int = 4
    device: t.Literal["cpu", "cuda"] = "cpu"

    def __post_init__(self):
        self.device = device_check(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

        model_config = self.model.config.to_dict()
        assert model_config.get("id2label") or model_config.get(
            "label2id"
        ), "label-id mapping missing"
        if model_config.get("id2label") is None:
            self.id2label = {v: k for k, v in model_config.label2id}
        else:
            self.id2label = model_config["id2label"]

    @property
    def name(self):
        return "Entailment_score"

    @property
    def is_batchable(self):
        return True

    def batch_infer(self, inputs: dict):
        predictions = []
        input_ids = inputs["input_ids"]
        label2id = {value.lower(): key for key, value in self.id2label.items()}

        for idx in range(0, len(input_ids), self.batch_size):
            batch_ids = input_ids[idx : idx + self.batch_size]
            output = self.model(batch_ids.to(self.device))
            pred = output.logits.softmax(axis=-1).detach().cpu()
            predictions.extend(pred[:, label2id["entailment"]].tolist())

        return predictions

    def score(
        self,
        ground_truth: t.List[str],
        generated_text: t.List[str],
    ):
        """
        ground_truth : premis
        generated_text : hypothesis
        returns entailement probability score
        """

        encodings = self.tokenizer(
            ground_truth,
            generated_text,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
        )

        score = self.batch_infer(encodings)

        return score
