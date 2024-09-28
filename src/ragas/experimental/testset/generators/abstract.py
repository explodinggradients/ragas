import logging
import math
import random
import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import SingleTurnSample
from ragas.executor import run_async_batch
from ragas.experimental.prompt import PydanticPrompt, StringIO
from ragas.experimental.testset.generators.base import (
    BaseSimulator,
    BasicScenario,
    UserInputLength,
    UserInputStyle,
)
from ragas.experimental.testset.generators.prompts import (
    AbstractQuestionFromTheme,
    CommonThemeFromSummaries,
    CriticUserInput,
    GenerateReference,
    ModifyUserInput,
    Summaries,
    ThemeAndContext,
    Themes,
    UserInputAndContext,
    UserInputWithStyleAndLength,
    extend_modify_input_prompt,
)
from ragas.experimental.testset.graph import KnowledgeGraph, NodeType

logger = logging.getLogger(__name__)


class AbstractQuestionScenario(BasicScenario):
    theme: str


@dataclass
class AbstractQuestionSimulator(BaseSimulator):
    generate_user_input_prompt: PydanticPrompt = field(
        default_factory=AbstractQuestionFromTheme
    )
    critic_user_input_prompt: PydanticPrompt = field(default_factory=CriticUserInput)
    user_input_modification_prompt: PydanticPrompt = field(
        default_factory=ModifyUserInput
    )
    generate_reference_prompt: PydanticPrompt = field(default_factory=GenerateReference)

    def __post_init__(self):
        self.common_theme_prompt = CommonThemeFromSummaries()

    async def generate_scenarios(
        self, n: int, knowledge_graph: KnowledgeGraph
    ) -> t.List[AbstractQuestionScenario]:
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
            kw_list.append({"data": summaries, "llm": self.llm})

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

        # sample question styles and question lengths
        question_styles = random.choices(
            list(UserInputStyle), k=num_clusters * num_themes
        )
        question_lengths = random.choices(
            list(UserInputLength), k=num_clusters * num_themes
        )

        # create distributions
        distributions = []
        for cluster, theme, style, length in zip(
            clusters_sampled, themes_sampled, question_styles, question_lengths
        ):
            distributions.append(
                AbstractQuestionScenario(
                    theme=theme.theme,
                    nodes=cluster,
                    style=style,
                    length=length,
                )
            )
        return distributions

    async def generate_sample(
        self, scenario: AbstractQuestionScenario
    ) -> SingleTurnSample:
        user_input = await self.generate_user_input(scenario)
        if await self.critic_user_input(user_input):
            user_input = await self.modify_user_input(user_input, scenario)

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

    async def generate_user_input(self, scenario: AbstractQuestionScenario) -> str:
        question = await self.generate_user_input_prompt.generate(
            data=ThemeAndContext(
                theme=scenario.theme,
                context=self.make_source_text(scenario),
            ),
            llm=self.llm,
        )
        return question.text

    async def critic_user_input(self, user_input: str) -> bool:
        critic = await self.critic_user_input_prompt.generate(
            data=StringIO(text=user_input), llm=self.llm
        )
        return critic.independence > 1 and critic.clear_intent > 1

    async def modify_user_input(
        self, user_input: str, scenario: AbstractQuestionScenario
    ) -> str:
        prompt = extend_modify_input_prompt(
            question_modification_prompt=self.user_input_modification_prompt,
            style=scenario.style,
            length=scenario.length,
        )
        modified_question = await prompt.generate(
            data=UserInputWithStyleAndLength(
                user_input=user_input,
                style=scenario.style,
                length=scenario.length,
            ),
            llm=self.llm,
        )
        return modified_question.text

    async def generate_reference(
        self, user_input: str, scenario: AbstractQuestionScenario
    ) -> str:
        reference = await self.generate_reference_prompt.generate(
            data=UserInputAndContext(
                user_input=user_input,
                context=self.make_source_text(scenario),
            ),
            llm=self.llm,
        )
        return reference.text
