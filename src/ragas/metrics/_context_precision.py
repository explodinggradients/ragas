from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group

from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.utils import json_loader

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

CONTEXT_PRECISION = Prompt(
    name="context_precision",
    instruction="""Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not. """,
    examples=[
        {
            "question": """What can you tell me about albert Albert Einstein?""",
            "context": """Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.""",
            "answer": """Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895""",
            "verification": """{
                "reason": "The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.",
                "Verdict": "1"
            }
            """,
        },
        {
            "question": """who won 2020 icc world cup?""",
            "context": """Who won the 2022 ICC Men's T20 World Cup?""",
            "answer": """England""",
            "verification": """{
                "reason": "the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",
                "verdict": "1"
            }
            """,
        },
        {
            "question": """What is the tallest mountain in the world?""",
            "context": """The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.""",
            "answer": """Mount Everest.""",
            "verification": """{
                "reason":"the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.",
                "verdict":"0"
            }
            """,
        },
    ],
    input_keys=["question", "context", "answer"],
    output_key="verification",
    output_type="json",
)


@dataclass
class ContextPrecision(MetricWithLLM):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_precision"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qcg  # type: ignore
    context_precision_prompt: Prompt = field(default_factory=lambda: CONTEXT_PRECISION)
    batch_size: int = 15

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        logging.info(f"Adapting Context Precision to {language}")
        self.context_precision_prompt = self.context_precision_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.context_precision_prompt.save(cache_dir)

    def get_dataset_attributes(self, dataset: Dataset):
        answer = "ground_truths"
        if answer not in dataset.features.keys():
            logging.warning(
                "Using 'context_precision' without ground truth will be soon depreciated. Use 'context_utilization' instead"
            )
            answer = "answer"

        return dataset["question"], dataset["contexts"], dataset[answer]

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[Callbacks] = None,
        callback_group_name: str = "batch",
    ) -> list:
        prompts = []
        questions, contexts, answers = self.get_dataset_attributes(dataset)

        cb = CallbackManager.configure(inheritable_callbacks=callbacks)
        with trace_as_chain_group(
            callback_group_name, callback_manager=cb
        ) as batch_group:
            for qstn, ctx, answer in zip(questions, contexts, answers):
                human_prompts = [
                    self.context_precision_prompt.format(
                        question=qstn, context=c, answer=answer
                    )
                    for c in ctx
                ]

                prompts.extend(human_prompts)

            responses: list[list[str]] = []
            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )
            responses = [[i.text for i in r] for r in results.generations]
            context_lens = [len(ctx) for ctx in contexts]
            context_lens.insert(0, 0)
            context_lens = np.cumsum(context_lens)
            grouped_responses = [
                responses[start:end]
                for start, end in zip(context_lens[:-1], context_lens[1:])
            ]
            scores = []

            for response in grouped_responses:
                response = [
                    json_loader.safe_load(item, self.llm) for item in sum(response, [])
                ]
                response = [
                    int("1" == resp.get("verdict", "0").strip())
                    if resp.get("verdict")
                    else np.nan
                    for resp in response
                ]
                denominator = sum(response) + 1e-10
                numerator = sum(
                    [
                        (sum(response[: i + 1]) / (i + 1)) * response[i]
                        for i in range(len(response))
                    ]
                )
                scores.append(numerator / denominator)

        return scores


class ContextUtilization(ContextPrecision):
    name = "ContextUtilization"
    evaluation_mode = EvaluationMode.qac

    def get_dataset_attributes(self, dataset: Dataset):
        return dataset["question"], dataset["contexts"], dataset["answer"]


context_precision = ContextPrecision()
context_utilization = ContextUtilization()
