from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import typing as t

from langchain.prompts import ChatPromptTemplate
from numpy.random import default_rng

from ragas.llms import BaseRagasLLM
from ragas.llms.json_load import load_as_json
from ragas.llms.prompt import PromptValue
from ragas.testset.docstore import Document, DocumentStore
from ragas.testset.prompts import (
    FILTER_QUESTION,
    MULTICONTEXT_QUESTION,
    SCORE_CONTEXT,
    SEED_QUESTION,
    TABLE_QA,
    demonstrations,
)

rng = default_rng()
logger = logging.getLogger(__name__)


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


async def filter_node(
    llm: BaseRagasLLM, node: Document, threshold: float = 7.5
) -> t.Dict:
    """
    context: str
        The input context

    Checks if the context is has enough information to frame a question
    """
    human_prompt = SCORE_CONTEXT.format(context=node.page_content)
    prompt = ChatPromptTemplate.from_messages([human_prompt])
    results = await llm.agenerate_text(prompt=to_pv(prompt))
    output = results.generations[0][0].text.strip()
    score = load_as_json(output)
    # TODO: instead of updating score add a new "pass" key
    score.update({"score": score.get("score", 0) >= threshold})
    return score


async def filter_question(llm: BaseRagasLLM, question: str) -> bool:
    human_prompt = FILTER_QUESTION.format(question=question)
    prompt = ChatPromptTemplate.from_messages([human_prompt])

    results = await llm.agenerate_text(prompt=to_pv(prompt))
    results = results.generations[0][0].text.strip()
    json_results = load_as_json(results)
    logger.debug("filtered question: %s", json_results)
    return json_results.get("verdict") != "No"


@dataclass
class Evolution:
    def evolve(self):
        ...

    async def aevolve(self):
        ...


async def simple_evolution(
    llm: BaseRagasLLM, seed_doc: Document, is_table_present: bool = False
):
    if is_table_present:
        human_prompt = TABLE_QA.format(context=seed_doc.page_content)
    else:
        sample = rng.choice(demonstrations, 1)[0]  # type: ignore
        questions = rng.choice(sample["questions"], 2, replace=False)
        questions = (
            "{"
            + str({k: v for dic in questions.tolist() for k, v in dic.items()}).replace(
                "'", '"'
            )
            + "}"
        )
        demo = f'Context:{sample["context"]}\nQuestions:{questions}'
        human_prompt = SEED_QUESTION.format(
            demonstration=demo, context=seed_doc.page_content
        )

    prompt = ChatPromptTemplate.from_messages([human_prompt])
    results = llm.generate_text_with_hmpt(prompts=[prompt])
    results = results.generations[0][0].text
    if is_table_present:
        return [results]
    else:
        results = load_as_json(results)
        return [v for v in results.values()]


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
