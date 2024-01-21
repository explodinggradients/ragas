from __future__ import annotations

import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from fsspec.exceptions import asyncio
from numpy.random import default_rng

from ragas.llms import BaseRagasLLM
from ragas.llms.json_load import load_as_json
from ragas.testset.docstore import Direction, Document, DocumentStore, Node
from ragas.testset.prompts import (
    context_scoring_prompt,
    filter_question_prompt,
    multi_context_question_prompt,
    seed_question_prompt,
    compress_question_prompt,
    reasoning_question_prompt,
    evolution_elimination_prompt,
)

rng = default_rng()
logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from ragas.llms.prompt import Prompt


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
        prompt = context_scoring_prompt.format(context=node.page_content)
        results = await self.llm.agenerate_text(prompt=prompt)
        output = results.generations[0][0].text.strip()
        score = load_as_json(output)
        score.update({"score": score.get("score", 0) >= self.threshold})
        return score


@dataclass
class QuestionFilter(Filter):
    llm: BaseRagasLLM

    def filter(self, question: str) -> bool:
        return asyncio.get_event_loop().run_until_complete(self.afilter(question))

    async def afilter(self, question: str) -> bool:
        prompt = filter_question_prompt.format(question=question)
        results = await self.llm.agenerate_text(prompt=prompt)
        results = results.generations[0][0].text.strip()
        json_results = load_as_json(results)
        logger.debug("filtered question: %s", json_results)
        return json_results.get("verdict") != "No"


@dataclass
class EvolutionFilter(Filter):
    llm: BaseRagasLLM

    def filter(self, simple_question: str, compressed_question: str) -> bool:
        return asyncio.get_event_loop().run_until_complete(
            self.afilter(simple_question, compressed_question)
        )

    async def afilter(self, simple_question: str, compressed_question: str) -> bool:
        prompt = evolution_elimination_prompt.format(
            question1=simple_question, question2=compressed_question
        )
        results = await self.llm.agenerate_text(prompt=prompt)
        results = results.generations[0][0].text.strip()
        json_results = load_as_json(results)
        logger.debug("filtered question: %s", json_results)
        return json_results.get("verdict") != "No"


@dataclass
class CurrentNodes:
    root_node: Node
    nodes: t.List[Node] = field(default_factory=list)


@dataclass
class Evolution:
    generator_llm: BaseRagasLLM
    docstore: DocumentStore
    node_filter: NodeFilter
    question_filter: QuestionFilter
    max_tries: int = 5
    _tries: int = field(default=0, init=False, repr=False)

    @staticmethod
    def merge_nodes(nodes: CurrentNodes) -> Node:
        return Node(
            doc_id="merged", page_content=" ".join(n.page_content for n in nodes.nodes)
        )

    async def aretry_evolve(
        self, current_nodes: CurrentNodes, update_count: bool = True
    ):
        if update_count:
            self._tries += 1
        logger.info("retrying evolution: %s times", self._tries)
        if self._tries > self.max_tries:
            # TODO: make this into a custom exception
            raise ValueError("Max tries reached")
        return await self.aevolve(current_nodes)

    def _transform_question(self, prompt: Prompt, question: str) -> str:
        results = self.generator_llm.generate_text(
            prompt=prompt.format(question=question)
        )
        return results.generations[0][0].text.strip()

    @abstractmethod
    def evolve(self, current_nodes: CurrentNodes) -> str:
        ...

    @abstractmethod
    async def aevolve(self, current_nodes: CurrentNodes) -> str:
        ...


@dataclass
class ComplexEvolution(Evolution):
    se: SimpleEvolution = field(init=False, repr=False)
    evolution_filter: EvolutionFilter = field(init=False, repr=False)

    def __post_init__(self):
        # init simple evolution to get seed question
        self.se = SimpleEvolution(
            generator_llm=self.generator_llm,
            docstore=self.docstore,
            node_filter=self.node_filter,
            question_filter=self.question_filter,
        )
        # init evolution filter with critic llm from another filter
        self.evolution_filter = EvolutionFilter(self.node_filter.llm)


@dataclass
class SimpleEvolution(Evolution):
    def evolve(self, current_nodes: CurrentNodes) -> str:
        logger.info("evolving question")
        return asyncio.get_event_loop().run_until_complete(self.aevolve(current_nodes))

    def _get_more_adjacent_nodes(self, current_nodes: CurrentNodes):
        """
        if the evolutions doesn't have enough nodes to frame a question, get more nodes
        """
        # get more nodes from above the context window
        prev_adjacent_node = self.docstore.get_adjacent(
            current_nodes.nodes[0], Direction.PREV
        )
        if prev_adjacent_node is None:
            # get more nodes from below the context window
            next_adjacent_node = self.docstore.get_adjacent(
                current_nodes.nodes[-1], Direction.NEXT
            )
            if next_adjacent_node is not None:
                # add next nodes towards the end
                current_nodes.nodes.append(next_adjacent_node)
            else:
                # retry with new base node
                nodes = self.docstore.get_random_nodes(k=1)
                return CurrentNodes(root_node=nodes[0], nodes=nodes)
        else:
            # add prev nodes in index 0
            current_nodes.nodes.insert(0, prev_adjacent_node)

        return current_nodes

    async def aevolve(self, current_nodes: CurrentNodes) -> str:
        merged_node = self.merge_nodes(current_nodes)
        passed = await self.node_filter.afilter(current_nodes.root_node)
        if not passed["score"]:
            nodes = self.docstore.get_random_nodes(k=1)
            new_current_nodes = CurrentNodes(root_node=nodes[0], nodes=nodes)
            return await self.aretry_evolve(new_current_nodes, update_count=False)

        results = self.generator_llm.generate_text(
            prompt=seed_question_prompt.format(context=merged_node.page_content)
        )
        seed_question = results.generations[0][0].text
        # NOTE: might need improvement
        # select only one seed question here
        is_valid_question = await self.question_filter.afilter(seed_question)
        if not is_valid_question:
            # get more context to rewrite question
            current_nodes = self._get_more_adjacent_nodes(current_nodes)
            # retry with new nodes added
            return await self.aretry_evolve(current_nodes)
        else:
            # if valid question
            return seed_question


@dataclass
class MultiContextEvolution(ComplexEvolution):
    def evolve(self, current_nodes: CurrentNodes) -> str:
        logger.info("evolving question")
        return asyncio.get_event_loop().run_until_complete(self.aevolve(current_nodes))

    async def aevolve(self, current_nodes: CurrentNodes) -> str:
        simple_question = await self.se.aevolve(current_nodes)
        logger.debug(
            "[MultiContextEvolution] simple question generated: %s", simple_question
        )

        # find a similar node and generate a question based on both
        similar_node = self.docstore.get_similar(current_nodes.root_node)[0]
        prompt = multi_context_question_prompt.format(
            question=simple_question,
            context1=current_nodes.root_node.page_content,
            context2=similar_node,
        )
        results = await self.generator_llm.agenerate_text(prompt=prompt)
        question = results.generations[0][0].text.strip()
        logger.debug(
            "[MultiContextEvolution] multicontext question generated: %s", question
        )

        # compress the question
        compressed_question = self._transform_question(
            prompt=compress_question_prompt, question=question
        )
        logger.debug(
            "[MultiContextEvolution] multicontext question compressed: %s", question
        )

        if not await self.question_filter.afilter(compressed_question):
            # retry
            current_nodes = self.se._get_more_adjacent_nodes(current_nodes)
            return await self.aretry_evolve(current_nodes)

        assert self.evolution_filter is not None, "evolution filter cannot be None"
        if not await self.evolution_filter.afilter(
            simple_question, compressed_question
        ):
            # retry
            current_nodes = self.se._get_more_adjacent_nodes(current_nodes)
            return await self.aretry_evolve(current_nodes)

        return compressed_question


@dataclass
class ReasoningEvolution(ComplexEvolution):
    def evolve(self, current_nodes: CurrentNodes) -> str:
        logger.debug("evolving question")
        return asyncio.get_event_loop().run_until_complete(self.aevolve(current_nodes))

    async def aevolve(self, current_nodes: CurrentNodes) -> str:
        simple_question = await self.se.aevolve(current_nodes)
        logger.debug(
            "[ReasoningEvolution] simple question generated: %s", simple_question
        )

        result = await self.generator_llm.agenerate_text(
            prompt=reasoning_question_prompt.format(
                question=simple_question, context=current_nodes.root_node.page_content
            )
        )
        reasoning_question = result.generations[0][0].text.strip()
        #
        # compress the question
        compressed_question = self._transform_question(
            prompt=compress_question_prompt, question=reasoning_question
        )
        logger.debug(
            "[ReasoningEvolution] multicontext question compressed: %s",
            reasoning_question,
        )

        if not await self.question_filter.afilter(compressed_question):
            # retry
            current_nodes = self.se._get_more_adjacent_nodes(current_nodes)
            return await self.aretry_evolve(current_nodes)

        assert self.evolution_filter is not None, "evolution filter cannot be None"
        if not await self.evolution_filter.afilter(
            simple_question, compressed_question
        ):
            # retry
            current_nodes = self.se._get_more_adjacent_nodes(current_nodes)
            logger.debug(
                "evolution_filter failed, retrying with %s", len(current_nodes.nodes)
            )
            return await self.aretry_evolve(current_nodes)

        return reasoning_question
