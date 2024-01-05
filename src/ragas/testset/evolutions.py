from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate

from ragas.llms import RagasLLM
from ragas.testset.docstore import Document, DocumentStore
from ragas.testset.prompts import (
    FILTER_QUESTION,
    MULTICONTEXT_QUESTION,
    SCORE_CONTEXT,
    SEED_QUESTION,
)
from ragas.testset.testset_generator import load_as_score
from ragas.utils import load_as_json


@dataclass
class Filter(ABC):
    @abstractmethod
    def filter(self) -> bool:
        ...

    @abstractmethod
    async def afilter(self) -> bool:
        ...


async def filter_context(llm: RagasLLM, context: str, threshold: float = 7.5) -> bool:
    """
    context: str
        The input context

    Checks if the context is has enough information to frame a question
    """
    human_prompt = SCORE_CONTEXT.format(context=context)
    prompt = ChatPromptTemplate.from_messages([human_prompt])
    results = llm.generate(prompts=[prompt])
    output = results.generations[0][0].text.strip()
    score = load_as_score(output)
    return score >= threshold


def filter_question(llm: RagasLLM, question: str) -> bool:
    human_prompt = FILTER_QUESTION.format(question=question)
    prompt = ChatPromptTemplate.from_messages([human_prompt])

    results = llm.generate(prompts=[prompt])
    results = results.generations[0][0].text.strip()
    json_results = load_as_json(results)
    return json_results.get("verdict") != "No"


@dataclass
class Evolution:
    def evolve(self):
        ...

    async def aevolve(self):
        ...


def simple_evolution(llm: RagasLLM, seed_doc: Document):
    human_prompt = SEED_QUESTION.format(context=seed_doc.page_content)
    prompt = ChatPromptTemplate.from_messages([human_prompt])
    results = llm.generate(prompts=[prompt])
    question = results.generations[0][0].text.strip()
    return question


def multi_context_evolution(
    llm: RagasLLM, seed_doc: Document, doc_store: DocumentStore
):
    question = simple_evolution(llm, seed_doc)
    print(question)
    similar_context = doc_store.get_similar(seed_doc)[0]
    human_prompt = MULTICONTEXT_QUESTION.format(
        question=question, context1=seed_doc.page_content, context2=similar_context
    )
    prompt = ChatPromptTemplate.from_messages([human_prompt])
    results = llm.generate(prompts=[prompt])
    question = results.generations[0][0].text.strip()
    return question
