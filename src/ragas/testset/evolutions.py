from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import typing as t
from fsspec.exceptions import asyncio
from random import choice, choices

from langchain.prompts import ChatPromptTemplate
from numpy.random import default_rng

from ragas.llms import BaseRagasLLM
from ragas.llms.json_load import load_as_json
from ragas.llms.prompt import PromptValue
from ragas.testset.docstore import Document, DocumentStore, Direction
from ragas.testset.prompts import (
    FILTER_QUESTION,
    MULTICONTEXT_QUESTION,
    SCORE_CONTEXT,
    SEED_QUESTION,
    TABLE_QA,
    demonstrations,
)
from ragas.testset.docstore import Node

rng = default_rng()
logger = logging.getLogger(__name__)


def to_pv(prompt: ChatPromptTemplate) -> PromptValue:
    return PromptValue(prompt_str=prompt.format())


@dataclass
class Filter(ABC):
    ...


@dataclass
class NodeFilter(Filter):
    llm: BaseRagasLLM
    threshold: float = 7.5

    def filter(self, node: Node) -> t.Dict:
        return asyncio.get_event_loop().run_until_complete(self.afilter(node))

    async def afilter(self, node: Node) -> t.Dict:
        human_prompt = SCORE_CONTEXT.format(context=node.page_content)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = await self.llm.agenerate_text(prompt=to_pv(prompt))
        output = results.generations[0][0].text.strip()
        score = load_as_json(output)
        score.update({"score": score.get("score", 0) >= self.threshold})
        return score


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


@dataclass
class QuestionFilter(Filter):
    llm: BaseRagasLLM

    def filter(self, question: str) -> bool:
        return asyncio.get_event_loop().run_until_complete(self.afilter(question))

    async def afilter(self, question: str) -> bool:
        human_prompt = FILTER_QUESTION.format(question=question)
        prompt = ChatPromptTemplate.from_messages([human_prompt])

        results = await self.llm.agenerate_text(prompt=to_pv(prompt))
        results = results.generations[0][0].text.strip()
        json_results = load_as_json(results)
        logger.debug("filtered question: %s", json_results)
        return json_results.get("verdict") != "No"


@dataclass
class Evolution:
    ...


@dataclass
class SimpleEvolution(Evolution):
    node_filter: NodeFilter
    question_filter: QuestionFilter
    nodes: t.List[Node] = field(default_factory=list)
    max_ties: int = 5
    _tries: int = field(default=0, init=False, repr=False)

    def merged_nodes(self) -> Node:
        return Node(
            doc_id="merged", page_content=" ".join(n.page_content for n in self.nodes)
        )

    def evolve(self, llm: BaseRagasLLM, docstore: DocumentStore):
        logger.info("evolving question")
        return asyncio.get_event_loop().run_until_complete(self.aevolve(llm, docstore))

    async def aretry_evolve(
        self, llm: BaseRagasLLM, docstore: DocumentStore, update_count: bool = True
    ):
        if update_count:
            self._tries += 1
        print("retrying evolution: %s times", self._tries)
        if self._tries > self.max_ties:
            # TODO: make this into a custom exception
            raise ValueError("Max tries reached")
        return await self.aevolve(llm, docstore)

    async def aevolve(self, llm: BaseRagasLLM, docstore: DocumentStore):
        # can the node be used to frame a question?
        if self._tries == 0:
            self.nodes = docstore.get_random_nodes(k=1)
        merged_node = self.merged_nodes()
        passed, table_is_present = await self.node_filter.afilter(self.nodes[0])
        if not passed:
            self.nodes = docstore.get_random_nodes(k=1)
            return await self.aretry_evolve(llm, docstore, update_count=False)

        # frame a basic question with with node
        seed_questions = await simple_evolution(llm, merged_node, table_is_present)
        # NOTE: might need improvement
        # select only one seed question here
        seed_question = choice(seed_questions)
        is_valid_question = await self.question_filter.afilter(seed_question)
        if not is_valid_question:
            # get more context to rewrite question
            prev_adjacent_node = docstore.get_adjacent(self.nodes[0], Direction.PREV)
            if prev_adjacent_node is None:
                next_adjacent_node = docstore.get_adjacent(
                    self.nodes[-1], Direction.NEXT
                )
                if next_adjacent_node is not None:
                    # add nodes
                    self.nodes.append(next_adjacent_node)
            else:
                # add prev nodes
                self.nodes.insert(0, prev_adjacent_node)
            # retry with new nodes added
            return await self.aretry_evolve(llm, docstore)
        else:
            # if valid question
            return seed_question


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
    llm: BaseRagasLLM, seed_node: Node, doc_store: DocumentStore
):
    question = simple_evolution(llm, seed_node)
    print(question)
    similar_context = doc_store.get_similar(seed_node)[0]
    human_prompt = MULTICONTEXT_QUESTION.format(
        question=question, context1=seed_node.page_content, context2=similar_context
    )
    prompt = ChatPromptTemplate.from_messages([human_prompt])
    results = await llm.agenerate_text(prompt=to_pv(prompt))
    question = results.generations[0][0].text.strip()
    return question
