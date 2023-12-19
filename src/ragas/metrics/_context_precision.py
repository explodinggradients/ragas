from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.utils import json_loader

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

CONTEXT_PRECISION = HumanMessagePromptTemplate.from_template(
    """\
Please verify if the information in the given context is useful in answering the question. Here are guidelines to help you make the decision:

1. If the question has no answer, use your own judgement.
2. If the question has an answer, use both question and answer to make decision, give "Yes" verdict when all of the following conditions are met:
    a. Partial or complete answer statements can be obtained from the context.
    b. When you find useful information in the context, make sure the entity it is talking about is in the answer.

Use only "Yes" (1) or "No" (0) as a binary verdict. Output JSON with reason.

<question> What are the health benefits of green tea? </question>
<answer> None </answer>
<context> The article explores the history of tea in China, tracing its roots to ancient dynasties. It discusses regional tea varieties, brewing techniques, and the cultural significance of tea as a symbol of hospitality and relaxation. </context>
verification:
{{
    "reason":"The context, while informative about the history and cultural significance of tea in China, does not provide specific information about the health benefits of green tea. Thus, it is not useful for answering the question about health benefits.",
    "verdict":"0"
}}

<question> How does photosynthesis work in plants? </question>
<answer> None </answer>
<context> Photosynthesis in plants is a complex process where chlorophyll absorbs sunlight in chloroplasts, driving a chemical reaction converting carbon dioxide and water into glucose and oxygen. The process involves light and dark reactions, producing ATP and NADPH. </context>
verification:
{{
    "reason":"This context is extremely relevant and useful for answering the question. It directly addresses the mechanisms of photosynthesis, explaining the key components and processes involved.",
    "verdict":"1"
}}

<question> What factors should cancer patients consider in their dietary choices? </question>
<answer> Cancer patients need to avoid calcium supplementation. </answer>
<context> For cancer patients, the intake of refined sugar should be limited, because the starch in food can prevent colon and rectal cancer. High fiber diet may also prevent colon, rectal, breast cancer and pancreatic cancer cancer. </context>
verification:
{{
    reason: "The answer only mentions calcium supplementation, which is not addressed in the context. Therefore, the context is not useful for answering the question.",
    "verdict": "0"
}}

<question> Who was  Albert Einstein? </question>
<answer> He was a German-born theoretical physicist. </answer>
<context> Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. </context>
verification:
{{
    "reason": "In the context, Albert Einstein is described as a theoretical physicist who developed the theory of relativity. This is consistent with the answer, which states that he was a theoretical physicist. Therefore, the context is useful for answering the question.",
    "verdict": "1"
}}

<question> When did Qi Tian go to the United States? </question>
<question Qi Tian went to the United States in 1991. </question>
<context> in 1991, Qing Tian went to the United States. </context>
verification:
{{
    "reason": "Altough the context mentioned the year 1991, it did not mention the person Qi Tian. Therefore, the context is not useful for answering the question.",
    "verdict": "0"
}}

<question> {question} </question>
<answer> {answer} </answer>
<context> {context} </context>
verification:"""  # noqa: E501
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
    evaluation_mode: EvaluationMode = EvaluationMode.qc  # type: ignore
    batch_size: int = 15

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[Callbacks] = None,
        callback_group_name: str = "batch",
    ) -> list:
        prompts = []
        questions, contexts = dataset["question"], dataset["contexts"]
        if "ground_truths" in dataset.column_names:
            ground_truths = dataset["ground_truths"]
        else:
            ground_truths = [None] * len(questions)

        cb = CallbackManager.configure(inheritable_callbacks=callbacks)
        with trace_as_chain_group(
            callback_group_name, callback_manager=cb
        ) as batch_group:
            for qstn, ctx, gt in zip(questions, contexts, ground_truths):
                human_prompts = [
                    ChatPromptTemplate.from_messages(
                        [CONTEXT_PRECISION.format(question=qstn, context=c, answer=gt)]
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

            for q, resp in zip(questions, grouped_responses):
                print(q)
                print(resp)

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


context_precision = ContextPrecision()
