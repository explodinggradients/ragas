from __future__ import annotations
from textwrap import indent

import json
import numpy as np
import spacy
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import typing as t
from dataclasses import dataclass

import transformers

from belar.metrics import Metric
from belar.utils import device_check

SPACY_MODEL = "en_core_web_sm"

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
            ground_truth, generated_text, truncation=True, return_tensors="pt",
            max_length=self.max_length, padding="max_length",
        )

        score = self.batch_infer(encodings)

        return score


@dataclass
class Qsquare(Metric):

    qa_model_name:str = ""
    qg_model_name: str = ""
    device: t.Literal["cpu", "cuda"] = "cpu"
    max_answers: int = 10
    candidate_filter: bool = True
    load_single = False
    batch_size: int = 4


    def __post_init__(self,):
        
        self.nlp = spacy.load(SPACY_MODEL)
        self.qa = QAGQ.from_pretrained(self.qa_model_name)
        self.qg = QAGQ.from_pretrained(self.qg_model_name)
        


    def generate_candidates(self, text:str):

        text = text.strip()
        nouns = [i.text.lower() for i in self.nlp(text).noun_chunks if i.text.lower() not in self.nlp.Defaults.stop_words]
        entities = [ent.text.lower() for ent in self.nlp(text)]
        num_nouns = max(0, self.max_answers - len(entities))
        nouns = np.random.choice(np.setdiff1d(nouns, entities), size=num_nouns).tolist()

        return entities + nouns
    
    def generate_questions(self,candidates:list[str], context:str, **kwargs):

        questions = []
        for idx in range(0, len(candidates), self.batch_size):
            batch_questions = self.qg.batch_generate_question(candidates[idx:idx+self.batch_size],
                                           context, **kwargs)
            questions.extend(batch_questions)
        assert len(questions) == len(candidates), "Missing question for some candidates"
        return questions

    def generate_answers(self, questions: list[str], context:str):

        answers = []
        for idx in range(0, len(questions), self.batch_size):
            batch_answers = self.qa.batch_generate_answers(questions[idx:idx+self.batch_size],context)
            answers.extend(batch_answers)
        assert len(answers) == len(questions), "Missing answers for some questions"
        return answers

    def filter_candidates(self, candidates:list[str],context:str):
        
        pass

        
    def score(self, ground_truth: list[str], generated_text: list[str]):
        
        gnd_qans = {}
        ans_candidates = [self.generate_candidates(text) for text in ground_truth]
        for i,(candidates, context) in enumerate(zip(ans_candidates, ground_truth)):
            questions = self.generate_questions(candidates, context, **kwargs)
            gnd_qans[i] = [{"question": qstn, "answer":ans} for qstn, ans in zip(questions, candidates)]

        for i,gen_text in enumerate(generated_text):
            questions = [item["question"] for item in gnd_qans[i]]
            gen_answers = self.generate_answers(questions, gen_text)
            gnd_qans[i] = [item.update({"predicted_answer":ans}) for item, ans in zip(gnd_qans[i], gen_answers)]

        with open("qa-qj-intermediate.json", "w") as file:
            json.dump(gnd_qans, file, indent=4)



from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (
    MODEL_WITH_LM_HEAD_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)

MODEL_MAPPINGS_NAMES = [MODEL_WITH_LM_HEAD_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES]


class QAGQ:

    def __init__(self,model:PreTrainedModel, model_name_or_path:str, device="cpu"):

        self.model = model.from_pretrained(model_name_or_path)
        self.model.eval()
        self.device = device_check(device)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    @classmethod
    def from_pretrained(cls, model_name_or_path):

        config = AutoConfig.from_pretrained(model_name_or_path)
        model_mappings = [arch for model_type in MODEL_MAPPINGS_NAMES for arch in model_type.values()]
        architecture = np.intersect1d(model_mappings, config.architectures)
        if len(architecture) == 0:
            raise ValueError("Model doesn't support QA or LM architecture")
        model = getattr(transformers, architecture[0])
        return cls(model, model_name_or_path)

    
    def batch_generate_question(self,answers:list[str], contexts:str,**kwargs):

        input_texts = ["answer: %s  context: %s </s>" % (ans, context) for ans in answers]
        max_length = kwargs.pop("input_max_length",512)
        encodings = self.tokenizer(input_texts, padding="max_length", truncation=True,
                        max_length = max_length, return_tensors="pt")
        encodings = {k:v.to(self.device) for k,v in encodings.items()}
        outputs = self.model.generate(**encodings,**kwargs)
        outputs = self.tokenizer.batch_decode(outputs,skip_special_tokens=True)
        return [output.replace("question:","").strip() for output in outputs]

    def batch_generate_answers(self, questions:list[str], context:str, **kwargs):
        
        max_length = kwargs.pop("input_max_length",512)
        encodings = self.tokenizer(questions, [context] * len(questions), padding="max_length", truncation=True,
                        max_length = max_length, return_tensors="pt")
        encodings = {k:v.view(-1,max_length).to(self.device) for k,v in encodings.items()}
        poss_ans_starts, poss_ans_ends = self.model(**encodings, return_dict=False)
        best_start = poss_ans_starts.argmax(1)
        best_ends = poss_ans_ends.argmax(1)
        answers = [encodings["input_ids"][i][start:end+1] for i, (start, end) in enumerate(zip(best_start, best_ends))]
        answers = self.tokenizer.batch_decode(answers)
        return answers
