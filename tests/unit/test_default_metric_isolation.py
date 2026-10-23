"""Offline integration coverage for concurrent default-metric evaluations."""

import asyncio
import socket

import pytest

from ragas import EvaluationDataset, aevaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig


class FakeEmbedding(BaseRagasEmbeddings):
    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


class FakeLLM(InstructorBaseRagasLLM):
    is_async = True

    def __init__(self, label, entered, release, calls):
        self.label = label
        self.entered = entered
        self.release = release
        self.calls = calls

    def generate(self, prompt, response_model):
        raise AssertionError("Only asynchronous generation expected")

    async def agenerate(self, prompt, response_model):
        run = "B" if "RUN_B" in prompt else "A"
        kind = response_model.__name__
        self.calls.append([run, self.label, kind])
        if kind == "StatementGeneratorOutput":
            self.entered[run].set()
            await self.release[run].wait()
        verdict = int(self.label == "A")
        responses = {
            "StatementGeneratorOutput": {"statements": [f"RUN_{run} statement"]},
            "NLIStatementOutput": {
                "statements": [
                    {
                        "statement": f"RUN_{run} statement",
                        "reason": "offline fixture",
                        "verdict": verdict,
                    }
                ]
            },
            "ResponseRelevanceOutput": {
                "question": f"RUN_{run} question",
                "noncommittal": 0,
            },
            "Verification": {"reason": "offline fixture", "verdict": verdict},
            "ContextRecallClassifications": {
                "classifications": [
                    {
                        "statement": f"RUN_{run} statement",
                        "reason": "offline fixture",
                        "attributed": verdict,
                    }
                ]
            },
        }
        return response_model(**responses[kind])


def dataset(label):
    return EvaluationDataset.from_list(
        [
            {
                "user_input": f"RUN_{label} question",
                "response": f"RUN_{label} answer",
                "retrieved_contexts": [f"RUN_{label} context"],
                "reference": f"RUN_{label} reference",
            }
        ]
    )


def independent_metrics():
    return [AnswerRelevancy(), ContextPrecision(), Faithfulness(), ContextRecall()]


async def trial(isolated, reset):
    entered = {label: asyncio.Event() for label in "AB"}
    release = {label: asyncio.Event() for label in "AB"}
    calls = []
    llms = {label: FakeLLM(label, entered, release, calls) for label in "AB"}

    async def evaluate(label):
        kwargs = {"metrics": independent_metrics()} if isolated else {}
        return await aevaluate(
            dataset(label),
            llm=llms[label],
            embeddings=FakeEmbedding(),
            show_progress=False,
            raise_exceptions=True,
            run_config=RunConfig(timeout=5, max_retries=1),
            **kwargs,
        )

    tasks = []
    try:
        a = asyncio.create_task(evaluate("A"))
        tasks.append(a)
        await entered["A"].wait()
        b = asyncio.create_task(evaluate("B"))
        tasks.append(b)
        await entered["B"].wait()
        if reset:
            release["A"].set()
            result_a = await a
            release["B"].set()
            try:
                result_b = await b
                outcome_b = result_b.scores[0]["faithfulness"]
            except AssertionError as error:
                outcome_b = f"{type(error).__name__}: {error}"
        else:
            release["B"].set()
            result_b = await b
            outcome_b = result_b.scores[0]["faithfulness"]
            release["A"].set()
            result_a = await a
        return {
            "isolated_instances": isolated,
            "A_finishes_first": reset,
            "A_faithfulness": result_a.scores[0]["faithfulness"],
            "B_faithfulness": outcome_b,
            "B_provider_labels": sorted(
                set(provider for run, provider, _ in calls if run == "B")
            ),
            "calls": calls,
        }

    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden in offline evaluation test")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "isolated", [False, True], ids=["default-metrics", "independent-control"]
)
@pytest.mark.parametrize("reset", [False, True], ids=["B-first", "A-first"])
async def test_evaluation_uses_its_own_provider(isolated, reset):
    result = await asyncio.wait_for(trial(isolated, reset), timeout=15)
    assert result["A_faithfulness"] == 1.0
    assert result["B_faithfulness"] == 0.0
    assert result["B_provider_labels"] == ["B"]


@pytest.mark.asyncio
async def test_explicit_metric_keeps_its_configured_llm():
    entered = {label: asyncio.Event() for label in "AB"}
    release = {label: asyncio.Event() for label in "AB"}
    for event in release.values():
        event.set()
    calls = []
    configured = FakeLLM("A", entered, release, calls)
    fallback = FakeLLM("B", entered, release, calls)
    metric = Faithfulness(llm=configured)
    result = await asyncio.wait_for(
        aevaluate(
            dataset("A"),
            metrics=[metric],
            llm=fallback,
            show_progress=False,
            raise_exceptions=True,
        ),
        timeout=15,
    )
    assert result.scores[0]["faithfulness"] == 1.0
    assert metric.llm is configured
    assert {provider for _, provider, _ in calls} == {"A"}


@pytest.mark.asyncio
async def test_default_metrics_match_explicit_metrics_sequentially():
    entered = {label: asyncio.Event() for label in "AB"}
    release = {label: asyncio.Event() for label in "AB"}
    for event in release.values():
        event.set()
    llm = FakeLLM("A", entered, release, [])
    results = []
    for kwargs in ({}, {"metrics": independent_metrics()}):
        result = await asyncio.wait_for(
            aevaluate(
                dataset("A"),
                llm=llm,
                embeddings=FakeEmbedding(),
                show_progress=False,
                raise_exceptions=True,
                **kwargs,
            ),
            timeout=15,
        )
        results.append(result.scores)
    assert list(results[0][0]) == [
        "answer_relevancy",
        "context_precision",
        "faithfulness",
        "context_recall",
    ]
    assert results[0] == results[1]
