from __future__ import annotations

import json
import re
import string
import typing as t
from dataclasses import dataclass
from warnings import warn

import numpy as np
import spacy
import torch
import transformers
from spacy.cli.download import download as spacy_download
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

from ragas.metrics import Metric
from ragas.utils import device_check

if t.TYPE_CHECKING:
    from torch import device as Device

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_WITH_LM_HEAD_MAPPING_NAMES,
)

MODEL_MAPPINGS_NAMES = [
    MODEL_WITH_LM_HEAD_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
]

DEVICES = ["cpu", "cuda"]
SPACY_MODEL = "en_core_web_sm"
LABEL2SCORE = {"entailment": 1, "contradiction": 0, "neutral": 0.5}
EPS = 1e-8


@dataclass
class EntailmentScore(Metric):
    """
    Entailment score using ground truth as premise and generated text as hypothesis.
    """

    model_name: str = "typeform/distilbert-base-uncased-mnli"
    max_length: int = 512
    batch_size: int = 4
    device: t.Literal["cpu", "cuda"] | Device = "cpu"

    def init_model(self):
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

    def infer(self, ground_truth: str, generated_text: str):
        encodings = self.tokenizer(
            ground_truth,
            generated_text,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
        )
        label2id = {value.lower(): key for key, value in self.id2label.items()}
        output = self.model(**encodings)
        pred = output.logits.softmax(axis=-1).detach().cpu().squeeze()
        return {label: pred[id].item() for label, id in label2id.items()}

    def batch_infer(self, inputs: dict):
        predictions = []
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        label2id = {value.lower(): key for key, value in self.id2label.items()}

        for idx in range(0, len(input_ids), self.batch_size):
            batch_inp_ids = input_ids[idx : idx + self.batch_size]
            batch_attn_mask = attention_mask[idx : idx + self.batch_size]

            output = self.model(
                batch_inp_ids.to(self.device), batch_attn_mask.to(self.device)
            )
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


class QAGQ:
    def __init__(
        self,
        model: PreTrainedModel,
        model_name_or_path: str,
        device: t.Literal["cpu", "cuda"] | Device = "cpu",
    ):
        self.model = model.from_pretrained(model_name_or_path)
        self.model.eval()  # type: ignore
        self.device = device_check(device)
        self.model.to(self.device)  # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path)
        model_mappings = [
            arch for model_type in MODEL_MAPPINGS_NAMES for arch in model_type.values()
        ]
        architecture = np.intersect1d(model_mappings, config.architectures)
        if len(architecture) == 0:
            raise ValueError("Model doesn't support QA or LM architecture")
        model = getattr(transformers, architecture[0])
        return cls(model, model_name_or_path)

    def batch_generate_question(self, answers: list[str], context: str, **kwargs):
        input_texts = [
            "answer: %s  context: %s </s>" % (ans, context) for ans in answers
        ]
        max_length = kwargs.pop("input_max_length", 512)
        encodings = self.tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        outputs = self.model.generate(**encodings, **kwargs)  # type: ignore
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [output.replace("question:", "").strip() for output in outputs]

    def batch_generate_answers(self, questions: list[str], context: str, **kwargs):
        max_length = kwargs.pop("input_max_length", 512)
        encodings = self.tokenizer(
            questions,
            [context] * len(questions),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {
            k: v.view(-1, max_length).to(self.device) for k, v in encodings.items()
        }
        poss_ans_starts, poss_ans_ends = self.model(
            **encodings, return_dict=False
        )  # type: ignore
        best_start = poss_ans_starts.argmax(1)
        best_ends = poss_ans_ends.argmax(1)
        answers = [
            encodings["input_ids"][i][start : end + 1]
            for i, (start, end) in enumerate(zip(best_start, best_ends))
        ]
        answers = self.tokenizer.batch_decode(answers)
        return answers


@dataclass
class Qsquare(Metric):
    qa_model_name: str = "consciousAI/question-answering-roberta-base-s"
    qg_model_name: str = "mrm8488/t5-base-finetuned-question-generation-ap"
    device: t.Literal["cpu", "cuda"] = "cpu"
    max_answers: int = 10
    crosscheck_candidates: bool = True
    load_single = False
    batch_size: int = 4
    include_nouns: bool = True
    save_results: bool = False

    def init_model(self):
        self.qa = QAGQ.from_pretrained(self.qa_model_name)
        self.qg = QAGQ.from_pretrained(self.qg_model_name)
        self.nli = EntailmentScore()
        self.nli.init_model()
        try:
            self.nlp = spacy.load(SPACY_MODEL)
        except OSError:
            warn(
                f"Spacy model [{SPACY_MODEL}] not found. Please run "
                f"`python -m spacy download {SPACY_MODEL}` to install it."
            )
            # logger.warning(f"Spacy models '{spacy_model_name}' not found."
            # "  Downloading and installing.")
            spacy_download(SPACY_MODEL)
            self.nlp = spacy.load(SPACY_MODEL)

    @property
    def name(self):
        return "Qsquare"

    @property
    def is_batchable(self):
        return True

    def generate_candidates(self, text: str):
        text = text.strip()
        nouns = [
            i.text.lower()
            for i in self.nlp(text).noun_chunks
            if i.text.lower() not in self.nlp.Defaults.stop_words
        ]
        entities = set([ent.text.lower() for ent in self.nlp(text).ents])
        num_nouns = max(0, self.max_answers - len(entities))
        nouns = list(np.setdiff1d(nouns, list(entities)))
        if nouns and self.include_nouns:
            nouns = np.random.choice(nouns, size=num_nouns).tolist()
        else:
            nouns = []

        return list(entities.union(set(nouns)))

    def generate_questions(self, candidates: list[str], context: str, **kwargs):
        questions = []
        for idx in range(0, len(candidates), self.batch_size):
            batch_questions = self.qg.batch_generate_question(
                candidates[idx : idx + self.batch_size], context, **kwargs
            )
            questions.extend(
                [qstn if qstn.endswith("?") else f"{qstn}?" for qstn in batch_questions]
            )
        assert len(questions) == len(candidates), "Missing question for some candidates"
        return questions

    def generate_answers(self, questions: list[str], context: str):
        answers = []
        for idx in range(0, len(questions), self.batch_size):
            batch_answers = self.qa.batch_generate_answers(
                questions[idx : idx + self.batch_size], context
            )
            answers.extend(batch_answers)
        assert len(answers) == len(questions), "Missing answers for some questions"
        return answers

    def filter_candidates(
        self, questions: list[str], candidates: list[str], gen_answers: list[str]
    ):
        final_questions = []
        final_candidates = []
        for qstn, ans1, ans2 in zip(questions, candidates, gen_answers):
            if self.clean_candidate(ans1) == self.clean_candidate(ans2):
                final_candidates.append(ans1)
                final_questions.append(qstn)

        return final_questions, final_candidates

    def clean_candidate(self, text):
        text = text.strip().lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the|in|our)\b", " ", text)

        return text

    def score_candidates(self, ques_ans_dict: dict):
        for qas in ques_ans_dict.values():
            for item in qas:
                item["answer"] = self.clean_candidate(item["answer"])
                item["predicted_answer"] = self.clean_candidate(
                    item["predicted_answer"]
                )
                if item["answer"] == item["predicted_answer"]:
                    item.update({"score": 1})
                else:
                    qstn = item.get("question")
                    score_dict = self.nli.infer(
                        f'{qstn}{item.get("answer")}',
                        f'{qstn}{item.get("predicted_answer")}',
                    )
                    label = max(zip(score_dict.values(), score_dict.keys()))[1]
                    item.update({"score": LABEL2SCORE[label]})

        return ques_ans_dict

    def score(self, ground_truth: list[str], generated_text: list[str], **kwargs):
        gnd_qans = {}
        ans_candidates = [self.generate_candidates(text) for text in ground_truth]
        for i, (candidates, context) in enumerate(zip(ans_candidates, ground_truth)):
            questions = self.generate_questions(candidates, context, **kwargs)
            gen_answers = self.generate_answers(questions, context)
            if self.crosscheck_candidates:
                questions, candidates = self.filter_candidates(
                    questions, candidates, gen_answers
                )
            gnd_qans[i] = [
                {"question": qstn, "answer": ans}
                for qstn, ans in zip(questions, candidates)  # type: ignore
            ]

        for i, gen_text in enumerate(generated_text):
            questions = [item["question"] for item in gnd_qans[i]]
            gen_answers = self.generate_answers(questions, gen_text)
            _ = [
                item.update({"predicted_answer": ans})
                for item, ans in zip(gnd_qans[i], gen_answers)  # type: ignore
            ]

        # del self.qa
        # del self.qg

        gnd_qans = self.score_candidates(gnd_qans)

        if self.save_results:
            with open("qa-qj-intermediate.json", "w") as file:
                json.dump(gnd_qans, file, indent=4)

        scores = [[dic["score"] for dic in item] for item in gnd_qans.values()]
        scores = [sum(sublist) / (len(sublist) + EPS) for sublist in scores]
        return scores


device = "cuda" if torch.cuda.is_available() else "cpu"
entailment_score = EntailmentScore(device=device)
q_square = Qsquare(device=device)
