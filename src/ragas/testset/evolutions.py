from __future__ import annotations

import logging
import typing as t
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from ragas.exceptions import MaxRetriesExceeded
from ragas.llms import BaseRagasLLM
from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt
from ragas.run_config import RunConfig
from ragas.testset.docstore import Direction, DocumentStore, Node
from ragas.testset.filters import EvolutionFilter, NodeFilter, QuestionFilter
from ragas.testset.prompts import (
    compress_question_prompt,
    conditional_question_prompt,
    find_relevent_context_prompt,
    multi_context_question_prompt,
    question_answer_prompt,
    reasoning_question_prompt,
    seed_question_prompt,
)
from ragas.testset.utils import rng

logger = logging.getLogger(__name__)


@dataclass
class CurrentNodes:
    root_node: Node
    nodes: t.List[Node] = field(default_factory=list)


# (question, current_nodes, evolution_type)
EvolutionOutput = t.Tuple[str, CurrentNodes, str]


class DataRow(BaseModel):
    question: str
    contexts: t.List[str]
    ground_truth: str
    evolution_type: str


@dataclass
class Evolution:
    generator_llm: BaseRagasLLM = t.cast(BaseRagasLLM, None)
    docstore: t.Optional[DocumentStore] = None
    node_filter: t.Optional[NodeFilter] = None
    question_filter: t.Optional[QuestionFilter] = None
    question_answer_prompt: Prompt = field(
        default_factory=lambda: question_answer_prompt
    )
    find_relevent_context_prompt: Prompt = field(
        default_factory=lambda: find_relevent_context_prompt
    )
    max_tries: int = 5
    is_async: bool = True

    @staticmethod
    def merge_nodes(nodes: CurrentNodes) -> Node:
        return Node(
            doc_id="merged",
            page_content="\n".join(n.page_content for n in nodes.nodes),
            keyphrases=[phrase for n in nodes.nodes for phrase in n.keyphrases],
        )

    def init(self, is_async: bool = True, run_config: t.Optional[RunConfig] = None):
        self.is_async = is_async
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def set_run_config(self, run_config: RunConfig):
        if self.docstore:
            self.docstore.set_run_config(run_config)
        if self.generator_llm:
            self.generator_llm.set_run_config(run_config)
        if self.node_filter:
            self.node_filter.set_run_config(run_config)
        if self.question_filter:
            self.question_filter.set_run_config(run_config)

    async def aretry_evolve(
        self,
        current_tries: int,
        current_nodes: CurrentNodes,
        update_count: bool = True,
    ) -> EvolutionOutput:
        if update_count:
            current_tries += 1
        logger.info("retrying evolution: %s times", current_tries)
        if current_tries > self.max_tries:
            # TODO: make this into a custom exception
            raise MaxRetriesExceeded(self)
        return await self._aevolve(current_tries, current_nodes)

    async def _transform_question(self, prompt: Prompt, question: str) -> str:
        assert self.generator_llm is not None, "generator_llm cannot be None"

        results = await self.generator_llm.generate(
            prompt=prompt.format(question=question), is_async=self.is_async
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

    def _get_new_random_node(self):
        assert self.docstore is not None, "docstore cannot be None"
        new_node = self.docstore.get_random_nodes(k=1)[0]
        return CurrentNodes(root_node=new_node, nodes=[new_node])

    async def evolve(self, current_nodes: CurrentNodes) -> DataRow:
        # init tries with 0 when first called
        current_tries = 0
        (
            evolved_question,
            current_nodes,
            evolution_type,
        ) = await self._aevolve(current_tries, current_nodes)

        return await self.generate_datarow(
            question=evolved_question,
            current_nodes=current_nodes,
            evolution_type=evolution_type,
        )

    @abstractmethod
    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        ...

    async def generate_datarow(
        self,
        question: str,
        current_nodes: CurrentNodes,
        evolution_type: str,
    ):
        assert self.generator_llm is not None, "generator_llm cannot be None"

        node_content = [
            f"{i}\t{n.page_content}" for i, n in enumerate(current_nodes.nodes)
        ]
        results = await self.generator_llm.generate(
            prompt=self.find_relevent_context_prompt.format(
                question=question, contexts=node_content
            )
        )
        relevent_contexts_result = await json_loader.safe_load(
            results.generations[0][0].text.strip(), llm=self.generator_llm
        )
        relevant_context_indices = relevent_contexts_result.get(
            "relevant_context", None
        )
        if relevant_context_indices is None:
            relevant_context = CurrentNodes(
                root_node=current_nodes.root_node, nodes=current_nodes.nodes
            )
        else:
            relevant_context = current_nodes

        merged_nodes = self.merge_nodes(relevant_context)
        results = await self.generator_llm.generate(
            prompt=self.question_answer_prompt.format(
                question=question, context=merged_nodes.page_content
            )
        )
        answer = results.generations[0][0].text.strip()
        logger.debug("answer generated: %s", answer)

        if answer == "-1":
            answer = None

        return DataRow(
            question=question,
            contexts=[n.page_content for n in current_nodes.nodes],
            ground_truth="" if answer is None else answer,
            evolution_type=evolution_type,
        )

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the filter to a different language.
        """
        assert self.node_filter is not None, "node filter cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"

        self.question_answer_prompt = self.question_answer_prompt.adapt(
            language, self.generator_llm, cache_dir
        )
        self.find_relevent_context_prompt = self.find_relevent_context_prompt.adapt(
            language, self.generator_llm, cache_dir
        )
        self.node_filter.adapt(language, cache_dir)
        self.question_filter.adapt(language, cache_dir)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the filter prompts to a path.
        """
        assert self.node_filter is not None, "node filter cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"
        self.question_answer_prompt.save(cache_dir)
        self.find_relevent_context_prompt.save(cache_dir)
        self.node_filter.save(cache_dir)
        self.question_filter.save(cache_dir)


@dataclass
class SimpleEvolution(Evolution):
    seed_question_prompt: Prompt = field(default_factory=lambda: seed_question_prompt)

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        assert self.docstore is not None, "docstore cannot be None"
        assert self.node_filter is not None, "node filter cannot be None"
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"

        merged_node = self.merge_nodes(current_nodes)
        passed = await self.node_filter.filter(current_nodes.root_node)
        if not passed["score"]:
            nodes = self.docstore.get_random_nodes(k=1)
            new_current_nodes = CurrentNodes(root_node=nodes[0], nodes=nodes)
            return await self.aretry_evolve(
                current_tries, new_current_nodes, update_count=False
            )

        logger.debug("keyphrases in merged node: %s", merged_node.keyphrases)
        results = await self.generator_llm.generate(
            prompt=self.seed_question_prompt.format(
                context=merged_node.page_content,
                keyphrases=rng.choice(
                    np.array(merged_node.keyphrases), size=3
                ).tolist(),
            )
        )
        seed_question = results.generations[0][0].text
        # NOTE: might need improvement
        # select only one seed question here
        is_valid_question = await self.question_filter.filter(seed_question)
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

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.seed_question_prompt = self.seed_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        super().save(cache_dir)
        self.seed_question_prompt.save(cache_dir)


@dataclass
class ComplexEvolution(Evolution):
    se: t.Optional[SimpleEvolution] = field(default=None, repr=False)
    evolution_filter: t.Optional[EvolutionFilter] = field(default=None, repr=False)
    compress_question_prompt: Prompt = field(
        default_factory=lambda: compress_question_prompt
    )

    def init(self, is_async: bool = True, run_config: t.Optional[RunConfig] = None):
        if run_config is None:
            run_config = RunConfig()
        super().init(is_async=is_async, run_config=run_config)

        if self.se is None:
            # init simple evolution to get seed question
            self.se = SimpleEvolution(
                generator_llm=self.generator_llm,
                docstore=self.docstore,
                node_filter=self.node_filter,
                question_filter=self.question_filter,
            )
        # init evolution filter with critic llm from another filter
        assert self.node_filter is not None, "node filter cannot be None"
        if self.evolution_filter is None:
            self.evolution_filter = EvolutionFilter(self.node_filter.llm)

        # set run configs
        self.se.set_run_config(run_config)
        self.evolution_filter.set_run_config(run_config)

    async def _acomplex_evolution(
        self, current_tries: int, current_nodes: CurrentNodes, question_prompt: Prompt
    ):
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        simple_question, _, _ = await self.se._aevolve(current_tries, current_nodes)
        logger.debug(
            "[%s] simple question generated: %s",
            self.__class__.__name__,
            simple_question,
        )

        result = await self.generator_llm.generate(
            prompt=question_prompt.format(
                question=simple_question, context=current_nodes.root_node.page_content
            )
        )
        reasoning_question = result.generations[0][0].text.strip()

        # compress the question
        compressed_question = await self._transform_question(
            prompt=self.compress_question_prompt, question=reasoning_question
        )
        logger.debug(
            "[%s] multicontext question compressed: %s",
            self.__class__.__name__,
            reasoning_question,
        )

        if not await self.question_filter.filter(compressed_question):
            # retry
            current_nodes = self.se._get_more_adjacent_nodes(current_nodes)
            return await self.aretry_evolve(current_tries, current_nodes)

        assert self.evolution_filter is not None, "evolution filter cannot be None"
        if await self.evolution_filter.filter(simple_question, compressed_question):
            # retry
            current_nodes = self.se._get_new_random_node()
            logger.debug(
                "evolution_filter failed, retrying with %s", len(current_nodes.nodes)
            )
            return await self.aretry_evolve(current_tries, current_nodes)

        return reasoning_question, current_nodes

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.evolution_filter is not None, "evolution filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        super().adapt(language, cache_dir)
        self.se.adapt(language, cache_dir)
        self.compress_question_prompt = self.compress_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )
        self.evolution_filter.adapt(language, cache_dir)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        assert self.evolution_filter is not None, "evolution filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        super().save(cache_dir)
        self.se.save(cache_dir)
        self.evolution_filter.save(cache_dir)
        self.compress_question_prompt.save(cache_dir)


@dataclass
class MultiContextEvolution(ComplexEvolution):
    multi_context_question_prompt: Prompt = field(
        default_factory=lambda: multi_context_question_prompt
    )

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
        similar_node = self.docstore.get_similar(current_nodes.root_node)
        if similar_node == []:
            # retry
            current_nodes = self.se._get_more_adjacent_nodes(current_nodes)
            return await self.aretry_evolve(current_tries, current_nodes)

        prompt = self.multi_context_question_prompt.format(
            question=simple_question,
            context1=current_nodes.root_node.page_content,
            context2=similar_node,
        )
        results = await self.generator_llm.generate(prompt=prompt)
        question = results.generations[0][0].text.strip()
        logger.debug(
            "[MultiContextEvolution] multicontext question generated: %s", question
        )

        # compress the question
        compressed_question = await self._transform_question(
            prompt=self.compress_question_prompt, question=question
        )
        logger.debug(
            "[MultiContextEvolution] multicontext question compressed: %s", question
        )

        if not await self.question_filter.filter(compressed_question):
            # retry
            current_nodes = self.se._get_more_adjacent_nodes(current_nodes)
            return await self.aretry_evolve(current_tries, current_nodes)

        assert self.evolution_filter is not None, "evolution filter cannot be None"
        if await self.evolution_filter.filter(simple_question, compressed_question):
            # retry
            current_nodes = self.se._get_new_random_node()
            return await self.aretry_evolve(current_tries, current_nodes)

        return compressed_question, current_nodes, "multi_context"

    def __hash__(self):
        return hash(self.__class__.__name__)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.multi_context_question_prompt = self.multi_context_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        super().save(cache_dir)
        self.multi_context_question_prompt.save(cache_dir)


@dataclass
class ReasoningEvolution(ComplexEvolution):
    reasoning_question_prompt: Prompt = field(
        default_factory=lambda: reasoning_question_prompt
    )

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        result = await self._acomplex_evolution(
            current_tries, current_nodes, self.reasoning_question_prompt
        )
        return result[0], result[1], "reasoning"

    def __hash__(self):
        return hash(self.__class__.__name__)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.reasoning_question_prompt = self.reasoning_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        super().save(cache_dir)
        self.reasoning_question_prompt.save(cache_dir)


@dataclass
class ConditionalEvolution(ComplexEvolution):
    conditional_question_prompt: Prompt = field(
        default_factory=lambda: conditional_question_prompt
    )

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        result = await self._acomplex_evolution(
            current_tries, current_nodes, self.conditional_question_prompt
        )
        return result[0], result[1], "conditional"

    def __hash__(self):
        return hash(self.__class__.__name__)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.conditional_question_prompt = self.conditional_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        super().save(cache_dir)
        self.conditional_question_prompt.save(cache_dir)


simple = SimpleEvolution()
multi_context = MultiContextEvolution()
reasoning = ReasoningEvolution()
conditional = ConditionalEvolution()
