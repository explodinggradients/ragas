from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field
from string import Template

import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field
import asyncio
import os
import requests

from ragas.dataset_schema import SingleTurnSample
from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    ensembler,
    get_segmenter,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


class HasSegmentMethod(t.Protocol):
    def segment(self, text) -> t.Any:
        ...


logger = logging.getLogger(__name__)


class Statements(BaseModel):
    sentence_index: int = Field(
        ..., description="Index of the sentence from the statement list"
    )
    simpler_statements: t.List[str] = Field(..., description="the simpler statements")


class StatementsAnswers(BaseModel):
    __root__: t.List[Statements]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_statements_output_instructions = get_json_format_instructions(StatementsAnswers)
_statements_output_parser = RagasoutputParser(pydantic_object=StatementsAnswers)


LONG_FORM_ANSWER_PROMPT = Prompt(
    name="long_form_answer",
    output_format_instruction=_statements_output_instructions,
    instruction="Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON.",
    examples=[
        {
            "question": "Who was Albert Einstein and what is he best known for?",
            "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            "sentences": """
        0:He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. 
        1:He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
        """,
            "analysis": StatementsAnswers.parse_obj(
                [
                    {
                        "sentence_index": 0,
                        "simpler_statements": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                        ],
                    },
                    {
                        "sentence_index": 1,
                        "simpler_statements": [
                            "Albert Einstein was best known for developing the theory of relativity.",
                            "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
                        ],
                    },
                ]
            ).dicts(),
        }
    ],
    input_keys=["question", "answer", "sentences"],
    output_key="analysis",
    language="english",
)


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class StatementFaithfulnessAnswers(BaseModel):
    __root__: t.List[StatementFaithfulnessAnswer]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_faithfulness_output_instructions = get_json_format_instructions(
    StatementFaithfulnessAnswers
)
_faithfulness_output_parser = RagasoutputParser(
    pydantic_object=StatementFaithfulnessAnswers
)

NLI_STATEMENTS_MESSAGE = Prompt(
    name="nli_statements",
    instruction="Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.",
    output_format_instruction=_faithfulness_output_instructions,
    examples=[
        {
            "context": """John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
            "statements": [
                "John is majoring in Biology.",
                "John is taking a course on Artificial Intelligence.",
                "John is a dedicated student.",
                "John has a part-time job.",
            ],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "John is majoring in Biology.",
                        "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        "verdict": 0,
                    },
                    {
                        "statement": "John is taking a course on Artificial Intelligence.",
                        "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        "verdict": 0,
                    },
                    {
                        "statement": "John is a dedicated student.",
                        "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        "verdict": 1,
                    },
                    {
                        "statement": "John has a part-time job.",
                        "reason": "There is no information given in the context about John having a part-time job.",
                        "verdict": 0,
                    },
                ]
            ).dicts(),
        },
        {
            "context": """Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.""",
            "statements": ["Albert Einstein was a genius."],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "Albert Einstein was a genius.",
                        "reason": "The context and statement are unrelated",
                        "verdict": 0,
                    }
                ]
            ).dicts(),
        },
    ],
    input_keys=["context", "statements"],
    output_key="answer",
    output_type="json",
    language="english",
)  # noqa: E501


@dataclass
class Faithfulness(MetricWithLLM, SingleTurnMetric):
    name: str = "faithfulness"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts",
            }
        }
    )
    nli_statements_message: Prompt = field(
        default_factory=lambda: NLI_STATEMENTS_MESSAGE
    )
    statement_prompt: Prompt = field(default_factory=lambda: LONG_FORM_ANSWER_PROMPT)
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1
    _reproducibility: int = 1

    @property
    def reproducibility(self):
        return self._reproducibility

    @reproducibility.setter
    def reproducibility(self, value):
        if value < 1:
            logger.warning("reproducibility cannot be less than 1, setting to 1")
            value = 1
        elif value % 2 == 0:
            logger.warning(
                "reproducibility level cannot be set to even number, setting to odd"
            )
            value += 1
        self._reproducibility = value

    def __post_init__(self):
        if self.sentence_segmenter is None:
            language = self.nli_statements_message.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def _create_nli_prompt(self, row: t.Dict, statements: t.List[str]) -> PromptValue:
        assert self.llm is not None, "llm must be set to compute score"

        contexts = row["retrieved_contexts"]
        # check if the statements are support in the contexts
        contexts_str: str = "\n".join(contexts)
        statements_str: str = json.dumps(statements)
        prompt_value = self.nli_statements_message.format(
            context=contexts_str, statements=statements_str
        )
        return prompt_value

    def _create_statements_prompt(self, row: t.Dict) -> PromptValue:
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"

        text, question = row["response"], row["user_input"]
        sentences = self.sentence_segmenter.segment(text)
        sentences = [
            sentence for sentence in sentences if sentence.strip().endswith(".")
        ]
        sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(sentences)])
        prompt_value = self.statement_prompt.format(
            question=question, answer=text, sentences=sentences
        )
        return prompt_value

    def _compute_score(self, answers: StatementFaithfulnessAnswers):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.__root__
        )
        num_statements = len(answers.__root__)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        p_value = self._create_statements_prompt(row)
        statements = await self.llm.generate(
            p_value,
            callbacks=callbacks,
        )
        statements = await _statements_output_parser.aparse(
            statements.generations[0][0].text, p_value, self.llm, self.max_retries
        )

        if statements is None:
            return np.nan

        statements = [item["simpler_statements"] for item in statements.dicts()]
        statements = [item for sublist in statements for item in sublist]

        assert isinstance(statements, t.List), "statements must be a list"

        p_value = self._create_nli_prompt(row, statements)
        nli_result = await self.llm.generate(
            p_value,
            callbacks=callbacks,
            n=self._reproducibility,
        )

        nli_result_text = [
            nli_result.generations[0][i].text for i in range(self._reproducibility)
        ]
        faithfulness_list = [
            await _faithfulness_output_parser.aparse(
                text, p_value, self.llm, self.max_retries
            )
            for text in nli_result_text
        ]

        faithfulness_list = [
            faith.dicts() for faith in faithfulness_list if faith is not None
        ]

        if faithfulness_list:
            faithfulness_list = ensembler.from_discrete(
                faithfulness_list,
                "verdict",
            )

            faithfulness_list = StatementFaithfulnessAnswers.parse_obj(
                faithfulness_list
            )
        else:
            return np.nan

        return self._compute_score(faithfulness_list)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logger.info(f"Adapting Faithfulness metric to {language}")

        self.nli_statements_message = self.nli_statements_message.adapt(
            language, self.llm, cache_dir
        )
        self.statement_prompt = self.statement_prompt.adapt(
            language, self.llm, cache_dir
        )

        self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.nli_statements_message.save(cache_dir)
        self.statement_prompt.save(cache_dir)


MINICHECK_SYSTEM_PROMPT = (
    "Determine whether the provided claim is consistent with the "
    "corresponding document. Consistency in this context implies that all "
    "information presented in the claim is substantiated by the document. "
    "If not, it should be considered inconsistent. Please assess the "
    "claim's consistency with the document by responding with either \"Yes\" "
    "or \"No\"."
)
MINICHECK_USER_PROMPT_TEMPLATE = Template("Document: $document\nClaim: $claim")


@dataclass
class MiniCheckExample:
  document: str = ""
  claim: str = ""


@dataclass
class FaithfulnesswithMiniCheck(Faithfulness):
  name: str = "faithfulness_with_minicheck"  # type: ignore
  device: str = "cpu"
  batch_size: int = 10
  max_sequence_len: int = 10000  # max sequence can be 32768
  use_api: bool = False
  bespoke_api_key: str = ""
  max_concurrent_requests: int = 10

  def __post_init__(self):
    if self.use_api:
      self.bespoke_api_key = (self.bespoke_api_key if self.bespoke_api_key else
                              os.environ.get("BESPOKE_API_KEY"))
      if not self.bespoke_api_key:
        raise ValueError(
            f"No API key found for bespokelabs API. Please get your key "
            "at https://console.bespokelabs.ai, then provide it "
            "by passing the bespoke_api_key parameter to the "
            "constructor or set the BESPOKE_API_KEY environment variable.")
      self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
    else:
      try:
        import einops as einops
        import torch as torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
      except ImportError:
        raise ImportError(
            "einops, torch, and transformers must be installed to use this feature, "
            " try `pip install .[all]` to install the dependencies.")
      self._minicheck = AutoModelForCausalLM.from_pretrained(
          "bespokelabs/Bespoke-MiniCheck-7B", trust_remote_code=True
      )
      self._tokenizer = AutoTokenizer.from_pretrained(
          "bespokelabs/Bespoke-MiniCheck-7B")
      self._yes_tokens = []
      for token, token_id in self._tokenizer.get_vocab().items():
        if token.lower() == 'yes':
          self._yes_tokens.append(token_id)
      self._minicheck.to(self.device)
    super().__post_init__()

  def _create_examples(
      self, row: t.Dict, statements: t.List[str]
  ) -> t.List[str]:
    document = "\n".join(row["retrieved_contexts"])
    return [MiniCheckExample(document=document, claim=statement)
            for statement in statements]

  def _decode(self, prompts: t.List[str]):
    inputs = self._tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=self.max_sequence_len)
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    with torch.no_grad():
      outputs = self._minicheck.generate(
          **inputs,
          max_new_tokens=1,
          return_dict_in_generate=True,
          output_scores=True)

    return outputs

  def _extract_scores(self, outputs):
    logits = outputs.scores[0]
    probs = torch.softmax(logits, dim=-1)
    top_5_probs, top_5_indices = torch.topk(probs, 5, dim=-1)
    scores = []
    for i in range(logits.shape[0]):
      top_5_tokens = [
          self._tokenizer.decode(
              [idx]).lower() for idx in top_5_indices[i]]
      yes_prob = sum(
          prob for token,
          prob in zip(
              top_5_tokens,
              top_5_probs[i]) if token == 'yes')
      scores.append(int(yes_prob > 0.5))

    return scores

  def _score_examples_locally(
          self, examples: t.List[MiniCheckExample]) -> t.List[float]:
    prompts = []
    for example in examples:
      user_prompt = MINICHECK_USER_PROMPT_TEMPLATE.substitute(
          document=example.document, claim=example.claim)
      message = [
          {"role": "system", "content": MINICHECK_SYSTEM_PROMPT},
          {"role": "user", "content": user_prompt},
      ]
      prompt = self._tokenizer.apply_chat_template(
          message, add_generation_prompt=True, tokenize=False)
      prompts.append(prompt)
    scores = []
    for i in range(0, len(prompts), self.batch_size):
      logits = self._decode(prompts[i:i + self.batch_size])
      scores_batch = self._extract_scores(logits)
      scores.extend(scores_batch)
    return scores

  async def _score_examples_api(
          self,
          examples: t.List[MiniCheckExample]) -> t.List[float]:
    async def request_minicheck(example: MiniCheckExample) -> t.List[float]:
      def sync_request_minicheck(example: MiniCheckExample):
        try:
          response = requests.post(
              "https://api.bespokelabs.ai/v0/minicheck/factcheck",
              json={
                  "context": example.document,
                  "claim": example.claim
              },
              headers={"api_key": self.bespoke_api_key}
          )
          response.raise_for_status()
          return int(response.json()['support_prob'] > 0.5)
        except requests.RequestException as e:
          logger.warning(f"Bespoke API request failed: {str(e)}")
          return np.nan
      loop = asyncio.get_event_loop()
      return await loop.run_in_executor(
          None,
          sync_request_minicheck,
          example
      )
    return await asyncio.gather(*[
        request_minicheck(example) for example in examples])

  async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks) -> float:
    assert self.llm is not None, "LLM is not set"

    p_value = self._create_statements_prompt(row)
    statements = await self.llm.generate(
        p_value,
        callbacks=callbacks,
    )
    statements = await _statements_output_parser.aparse(
        statements.generations[0][0].text, p_value, self.llm, self.max_retries
    )

    if statements is None:
      return np.nan

    statements = [item["simpler_statements"] for item in statements.dicts()]
    statements = [item for sublist in statements for item in sublist]

    examples = self._create_examples(row, statements)
    if not self.use_api:
      scores = self._score_examples_locally(examples)
    else:
      scores = await self._score_examples_api(examples)
    return sum(scores) / len(scores)


faithfulness = Faithfulness()
