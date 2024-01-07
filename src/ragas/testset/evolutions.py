from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain.prompts import ChatPromptTemplate

from ragas.llms import BaseRagasLLM
from ragas.llms.json_load import load_as_json
from ragas.llms.prompt import PromptValue
from ragas.testset.docstore import Document, DocumentStore
from ragas.testset.prompts import (
    FILTER_QUESTION,
    MULTICONTEXT_QUESTION,
    SCORE_CONTEXT,
    SEED_QUESTION,
)
from ragas.testset.testset_generator import load_as_score


@dataclass
class Filter(ABC):
    @abstractmethod
    def filter(self) -> bool:
        ...

    @abstractmethod
    async def afilter(self) -> bool:
        ...


def to_pv(prompt: ChatPromptTemplate) -> PromptValue:
    return PromptValue(prompt_str=prompt.format())


async def filter_context(
    llm: BaseRagasLLM, context: str, threshold: float = 7.5
) -> bool:
    """
    context: str
        The input context

    Checks if the context is has enough information to frame a question
    """
    human_prompt = SCORE_CONTEXT.format(context=context)
    prompt = ChatPromptTemplate.from_messages([human_prompt])
    results = await llm.agenerate_text(prompt=to_pv(prompt))
    output = results.generations[0][0].text.strip()
    score = load_as_score(output)
    return score >= threshold


async def filter_question(llm: BaseRagasLLM, question: str) -> bool:
    human_prompt = FILTER_QUESTION.format(question=question)
    prompt = ChatPromptTemplate.from_messages([human_prompt])

    results = await llm.agenerate_text(prompt=to_pv(prompt))
    results = results.generations[0][0].text.strip()
    json_results = load_as_json(results)
    return json_results.get("verdict") != "No"


@dataclass
class Evolution:
    def evolve(self):
        ...

    async def aevolve(self):
        ...


async def simple_evolution(llm: BaseRagasLLM, seed_doc: Document):
    human_prompt = SEED_QUESTION.format(context=seed_doc.page_content)
    prompt = ChatPromptTemplate.from_messages([human_prompt])
    results = await llm.agenerate_text(prompt=to_pv(prompt))
    question = results.generations[0][0].text.strip()
    return question


async def multi_context_evolution(
    llm: BaseRagasLLM, seed_doc: Document, doc_store: DocumentStore
):
    question = simple_evolution(llm, seed_doc)
    print(question)
    similar_context = doc_store.get_similar(seed_doc)[0]
    human_prompt = MULTICONTEXT_QUESTION.format(
        question=question, context1=seed_doc.page_content, context2=similar_context
    )
    prompt = ChatPromptTemplate.from_messages([human_prompt])
    results = await llm.agenerate_text(prompt=to_pv(prompt))
    question = results.generations[0][0].text.strip()
    return question
