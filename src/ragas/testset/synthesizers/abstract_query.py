from __future__ import annotations

import logging
import math
import random
import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import SingleTurnSample
from ragas.executor import run_async_batch
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, NodeType

from .base import BaseScenario, QueryLength, QueryStyle
from .base_query import QuerySynthesizer
from .prompts import (
    AbstractQueryFromTheme,
    CAQInput,
    CommonConceptsFromKeyphrases,
    CommonThemeFromSummariesPrompt,
    ComparativeAbstractQuery,
    Concepts,
    KeyphrasesAndNumConcepts,
    Summaries,
    ThemeAndContext,
    Themes,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class AbstractQueryScenario(BaseScenario):
    theme: str


@dataclass
class AbstractQuerySynthesizer(QuerySynthesizer):
    generate_user_input_prompt: PydanticPrompt = field(
        default_factory=AbstractQueryFromTheme
    )

    def __post_init__(self):
        super().__post_init__()
        self.common_theme_prompt = CommonThemeFromSummariesPrompt()

    async def _generate_scenarios(
        self, n: int, knowledge_graph: KnowledgeGraph, callbacks: Callbacks
    ) -> t.List[AbstractQueryScenario]:
        node_clusters = knowledge_graph.find_clusters(
            relationship_condition=lambda rel: (
                True if rel.get_property("cosine_similarity") else False
            )
        )
        logger.info("found %d clusters", len(node_clusters))

        # filter out nodes that are not chunks
        node_clusters = [
            cluster
            for cluster in node_clusters
            if all(node.type == "chunk" for node in cluster)
        ]

        # find the number of themes to generation for given n and the num of clusters
        # will generate more themes just in case
        if len(node_clusters) == 0:
            node_clusters_new = []
            # if no clusters, use the nodes directly
            for node in knowledge_graph.nodes:
                if node.type == NodeType.CHUNK:
                    node_clusters_new.append([node])

            if len(node_clusters_new) == 0:
                raise ValueError(
                    "no clusters found. Try running a few transforms to populate the dataset"
                )
            node_clusters = node_clusters_new[:n]

        num_clusters = len(node_clusters)
        num_themes = math.ceil(n / num_clusters)
        logger.info("generating %d themes", num_clusters)

        kw_list = []
        for cluster in node_clusters:
            summaries = []
            for node in cluster:
                summary = node.get_property("summary")
                if summary is not None:
                    summaries.append(summary)

            summaries = Summaries(
                summaries=summaries,
                num_themes=num_themes,
            )
            kw_list.append({"data": summaries, "llm": self.llm, "callbacks": callbacks})

        themes: t.List[Themes] = run_async_batch(
            desc="Generating common themes",
            func=self.common_theme_prompt.generate,
            kwargs_list=kw_list,
        )

        # sample clusters and themes to get num_clusters * num_themes
        clusters_sampled = []
        themes_sampled = []
        themes_list = [theme.themes for theme in themes]
        for cluster, ts in zip(node_clusters, themes_list):
            for theme in ts:
                themes_sampled.append(theme)
                clusters_sampled.append(cluster)

        # sample query styles and query lengths
        query_styles = random.choices(list(QueryStyle), k=num_clusters * num_themes)
        query_lengths = random.choices(list(QueryLength), k=num_clusters * num_themes)

        # create distributions
        distributions = []
        for cluster, theme, style, length in zip(
            clusters_sampled, themes_sampled, query_styles, query_lengths
        ):
            distributions.append(
                AbstractQueryScenario(
                    theme=theme.theme,
                    nodes=cluster,
                    style=style,
                    length=length,
                )
            )
        return distributions

    async def _generate_sample(
        self, scenario: AbstractQueryScenario, callbacks: Callbacks
    ) -> SingleTurnSample:
        user_input = await self.generate_query(scenario, callbacks)
        if await self.critic_query(user_input):
            user_input = await self.modify_query(user_input, scenario, callbacks)

        reference = await self.generate_reference(user_input, scenario)

        reference_contexts = []
        for node in scenario.nodes:
            if node.get_property("page_content") is not None:
                reference_contexts.append(node.get_property("page_content"))

        return SingleTurnSample(
            user_input=user_input,
            reference=reference,
            reference_contexts=reference_contexts,
        )

    async def generate_query(
        self, scenario: AbstractQueryScenario, callbacks: Callbacks
    ) -> str:
        query = await self.generate_user_input_prompt.generate(
            data=ThemeAndContext(
                theme=scenario.theme,
                context=self.make_reference_contexts(scenario),
            ),
            llm=self.llm,
            callbacks=callbacks,
        )
        return query.text


class ComparativeAbstractQueryScenario(BaseScenario):
    common_concept: str


@dataclass
class ComparativeAbstractQuerySynthesizer(QuerySynthesizer):
    common_concepts_prompt: PydanticPrompt = field(
        default_factory=CommonConceptsFromKeyphrases
    )
    generate_query_prompt: PydanticPrompt = field(
        default_factory=ComparativeAbstractQuery
    )

    async def _generate_scenarios(
        self, n: int, knowledge_graph: KnowledgeGraph, callbacks: Callbacks
    ) -> t.List[ComparativeAbstractQueryScenario]:
        node_clusters = knowledge_graph.find_clusters(
            relationship_condition=lambda rel: (
                True if rel.get_property("summary_cosine_similarity") else False
            )
        )
        logger.info("found %d clusters", len(node_clusters))

        # find the number of themes to generation for given n and the num of clusters
        # will generate more themes just in case
        if len(node_clusters) == 0:
            node_clusters_new = []

            # if no clusters, use the nodes directly
            for node in knowledge_graph.nodes:
                if node.type == NodeType.DOCUMENT:
                    node_clusters_new.append([node])

            if len(node_clusters_new) == 0:
                raise ValueError(
                    "no clusters found. Try running a few transforms to populate the dataset"
                )
            node_clusters = node_clusters_new[:n]

        num_clusters = len(node_clusters)
        num_concepts = math.ceil(n / num_clusters)
        logger.info("generating %d common_themes", num_concepts)

        # generate common themes
        cluster_concepts = []
        kw_list: t.List[t.Dict] = []
        for cluster in node_clusters:
            keyphrases = []
            for node in cluster:
                keyphrases_node = node.get_property("keyphrases")
                if keyphrases_node is not None:
                    keyphrases.extend(keyphrases_node)

            kw_list.append(
                {
                    "data": KeyphrasesAndNumConcepts(
                        keyphrases=keyphrases,
                        num_concepts=num_concepts,
                    ),
                    "llm": self.llm,
                    "callbacks": callbacks,
                }
            )

        common_concepts: t.List[Concepts] = run_async_batch(
            desc="Generating common_concepts",
            func=self.common_concepts_prompt.generate,
            kwargs_list=kw_list,
        )

        # sample everything n times
        for cluster, common_concept in zip(node_clusters, common_concepts):
            for concept in common_concept.concepts:
                cluster_concepts.append((cluster, concept))

        query_lengths_sampled = random.choices(
            list(QueryLength), k=num_clusters * num_concepts
        )
        query_styles_sampled = random.choices(
            list(QueryStyle), k=num_clusters * num_concepts
        )
        logger.info(
            "len(query_lengths_sampled) = %d, len(query_styles_sampled) = %d, len(cluster_concepts) = %d",
            len(query_lengths_sampled),
            len(query_styles_sampled),
            len(cluster_concepts),
        )

        # make the scenarios
        scenarios = []
        for (cluster, concept), length, style in zip(
            cluster_concepts,
            query_lengths_sampled,
            query_styles_sampled,
        ):
            scenarios.append(
                ComparativeAbstractQueryScenario(
                    common_concept=concept,
                    nodes=cluster,
                    length=length,
                    style=style,
                )
            )
        return scenarios

    async def _generate_sample(
        self, scenario: ComparativeAbstractQueryScenario, callbacks: Callbacks
    ) -> SingleTurnSample:
        # generate the user input
        keyphrases = []
        summaries = []
        for n in scenario.nodes:
            keyphrases_node = n.get_property("keyphrases")
            if keyphrases_node is not None:
                keyphrases.extend(keyphrases_node)
            summary_node = n.get_property("summary")
            if summary_node is not None:
                summaries.append(summary_node)

        query = await self.generate_query_prompt.generate(
            data=CAQInput(
                concept=scenario.common_concept,
                keyphrases=keyphrases,
                summaries=summaries,
            ),
            llm=self.llm,
            callbacks=callbacks,
        )
        query = query.text

        # critic the query
        if not await self.critic_query(query):
            query = await self.modify_query(query, scenario, callbacks)

        # generate the answer
        answer = await self.generate_reference(
            query, scenario, callbacks, reference_property_name="summary"
        )

        # make the reference contexts
        # TODO: make this more efficient. Right now we are taking only the summary
        reference_contexts = []
        for node in scenario.nodes:
            if node.get_property("summary") is not None:
                reference_contexts.append(node.get_property("summary"))

        return SingleTurnSample(
            user_input=query,
            reference=answer,
            reference_contexts=reference_contexts,
        )
