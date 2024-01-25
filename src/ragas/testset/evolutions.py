from __future__ import annotations

import logging
import typing as t
from abc import abstractmethod
from dataclasses import dataclass, field

from fsspec.exceptions import asyncio
from langchain_core.pydantic_v1 import BaseModel
from numpy.random import default_rng

from ragas.llms import BaseRagasLLM
from ragas.llms.prompt import Prompt
from ragas.testset.docstore import Direction, DocumentStore, Node
from ragas.testset.filters import EvolutionFilter, NodeFilter, QuestionFilter
from ragas.testset.prompts import (
    compress_question_prompt,
    multi_context_question_prompt,
    question_answer_prompt,
    reasoning_question_prompt,
    seed_question_prompt,
)

rng = default_rng()
logger = logging.getLogger(__name__)


@dataclass
class CurrentNodes:
    root_node: Node
    nodes: t.List[Node] = field(default_factory=list)


# (question, current_nodes, evolution_type)
EvolutionOutput = t.Tuple[str, CurrentNodes, str]


class DataRow(BaseModel):
    question: str
    context: str
    answer: str
    evolution_type: str


@dataclass
class Evolution:
    generator_llm: BaseRagasLLM = t.cast(BaseRagasLLM, None)
    docstore: t.Optional[DocumentStore] = None
    node_filter: t.Optional[NodeFilter] = None
    question_filter: t.Optional[QuestionFilter] = None
    max_tries: int = 5

    @staticmethod
    def merge_nodes(nodes: CurrentNodes) -> Node:
        return Node(
            doc_id="merged", page_content="\n".join(n.page_content for n in nodes.nodes)
        )

    async def aretry_evolve(
        self, current_tries: int, current_nodes: CurrentNodes, update_count: bool = True
    ) -> EvolutionOutput:
        if update_count:
            current_tries += 1
        logger.info("retrying evolution: %s times", current_tries)
        if current_tries > self.max_tries:
            # TODO: make this into a custom exception
            raise ValueError("Max tries reached")
        return await self._aevolve(current_tries, current_nodes)

    def _transform_question(self, prompt: Prompt, question: str) -> str:
        assert self.generator_llm is not None, "generator_llm cannot be None"

        results = self.generator_llm.generate_text(
            prompt=prompt.format(question=question)
        )
        return results.generations[0][0].text.strip()

    def _get_more_adjacent_nodes(self, current_nodes: CurrentNodes):
        """
        if the evolutions doesn't have enough nodes to frame a question, get more nodes
        """
        assert self.docstore is not None, "docstore cannot be None"

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

    def evolve(self, current_nodes: CurrentNodes) -> DataRow:
        return asyncio.get_event_loop().run_until_complete(self.aevolve(current_nodes))

    async def aevolve(self, current_nodes: CurrentNodes) -> DataRow:
        # init tries with 0 when first called
        current_tries = 0
        (
            evolved_question,
            current_nodes,
            evolution_type,
        ) = await self._aevolve(current_tries, current_nodes)

        return self.generate_datarow(
            question=evolved_question,
            current_nodes=current_nodes,
            evolution_type=evolution_type,
        )

    @abstractmethod
    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        ...

    def generate_datarow(
        self,
        question: str,
        current_nodes: CurrentNodes,
        evolution_type: str,
    ):
        assert self.generator_llm is not None, "generator_llm cannot be None"

        merged_nodes = self.merge_nodes(current_nodes)
        results = self.generator_llm.generate_text(
            prompt=question_answer_prompt.format(
                question=question, context=merged_nodes.page_content
            )
        )
        answer = results.generations[0][0].text.strip()
        logger.debug("answer generated: %s", answer)

        if answer == "-1":
            answer = None

        return DataRow(
            question=question,
            context=merged_nodes.page_content,
            answer="" if answer is None else answer,
            evolution_type=evolution_type,
        )


@dataclass
class SimpleEvolution(Evolution):
    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        assert self.docstore is not None, "docstore cannot be None"
        assert self.node_filter is not None, "node filter cannot be None"
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"

        merged_node = self.merge_nodes(current_nodes)
        passed = await self.node_filter.afilter(current_nodes.root_node)
        if not passed["score"]:
            nodes = self.docstore.get_random_nodes(k=1)
            new_current_nodes = CurrentNodes(root_node=nodes[0], nodes=nodes)
            return await self.aretry_evolve(
                current_tries, new_current_nodes, update_count=False
            )

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
            return await self.aretry_evolve(current_tries, current_nodes)
        else:
            # if valid question
            return seed_question, current_nodes, "simple"

    def __hash__(self):
        return hash(self.__class__.__name__)


@dataclass
class ComplexEvolution(Evolution):
    se: t.Optional[SimpleEvolution] = field(default=None, repr=False)
    evolution_filter: t.Optional[EvolutionFilter] = field(default=None, repr=False)

    def init_evolution(self):
        # init simple evolution to get seed question
        self.se = SimpleEvolution(
            generator_llm=self.generator_llm,
            docstore=self.docstore,
            node_filter=self.node_filter,
            question_filter=self.question_filter,
        )
        # init evolution filter with critic llm from another filter
        assert self.node_filter is not None, "node filter cannot be None"
        self.evolution_filter = EvolutionFilter(self.node_filter.llm)


@dataclass
class MultiContextEvolution(ComplexEvolution):
    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        assert self.docstore is not None, "docstore cannot be None"
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        simple_question, _, _ = await self.se._aevolve(current_tries, current_nodes)
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
            return await self.aretry_evolve(current_tries, current_nodes)

        assert self.evolution_filter is not None, "evolution filter cannot be None"
        if not await self.evolution_filter.afilter(
            simple_question, compressed_question
        ):
            # retry
            current_nodes = self.se._get_more_adjacent_nodes(current_nodes)
            return await self.aretry_evolve(current_tries, current_nodes)

        return compressed_question, current_nodes, "multi_context"

    def __hash__(self):
        return hash(self.__class__.__name__)


@dataclass
class ReasoningEvolution(ComplexEvolution):
    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        simple_question, _, _ = await self.se._aevolve(current_tries, current_nodes)
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
            return await self.aretry_evolve(current_tries, current_nodes)

        assert self.evolution_filter is not None, "evolution filter cannot be None"
        if not await self.evolution_filter.afilter(
            simple_question, compressed_question
        ):
            # retry
            current_nodes = self.se._get_more_adjacent_nodes(current_nodes)
            logger.debug(
                "evolution_filter failed, retrying with %s", len(current_nodes.nodes)
            )
            return await self.aretry_evolve(current_tries, current_nodes)

        return reasoning_question, current_nodes, "reasoning"

    def __hash__(self):
        return hash(self.__class__.__name__)


simple = SimpleEvolution()
multi_context = MultiContextEvolution()
reasoning = ReasoningEvolution()
