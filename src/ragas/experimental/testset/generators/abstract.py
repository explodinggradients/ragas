import logging
import math
import random
import typing as t
from dataclasses import dataclass, field

from ragas.executor import run_async_batch
from ragas.experimental.prompt import PydanticPrompt, StringIO
from ragas.experimental.testset.generators.base import (
    BaseTestsetGenerator,
    BasicDistribution,
    UserInputLength,
    UserInputStyle,
    extend_modify_input_prompt,
)
from ragas.experimental.testset.generators.prompts import (
    AbstractQuestionFromTheme,
    CommonThemeFromSummaries,
    CriticQuestion,
    ModifyQuestion,
    QuestionWithStyleAndLength,
    Summaries,
    ThemeAndContext,
    Themes,
)
from ragas.experimental.testset.graph import KnowledgeGraph, Node

logger = logging.getLogger(__name__)


class AbstractQADistribution(BasicDistribution):
    theme: str


@dataclass
class AbstractGenerator(BaseTestsetGenerator):
    question_modification_prompt: PydanticPrompt = field(default_factory=ModifyQuestion)
    generate_question_prompt: PydanticPrompt = field(
        default_factory=AbstractQuestionFromTheme
    )
    critic_question_prompt: PydanticPrompt = field(default_factory=CriticQuestion)

    def __post_init__(self):
        self.common_theme_prompt = CommonThemeFromSummaries()

    async def generate_distributions(
        self, n: int, knowledge_graph: KnowledgeGraph
    ) -> t.List[AbstractQADistribution]:
        node_clusters = knowledge_graph.find_clusters(
            relationship_condition=lambda rel: True
            if rel.get_property("cosine_similarity")
            else False
        )
        logger.info("found %d clusters", len(node_clusters))

        # find the number of themes to generation for given n and the num of clusters
        # will generate more themes just in case
        num_clusters = len(node_clusters)
        num_themes = math.ceil(n / num_clusters)
        logger.info("generating %d themes", num_themes)

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
                AbstractQADistribution(
                    theme=theme.theme,
                    nodes=cluster,
                    style=style,
                    length=length,
                )
            )
        return distributions

    async def generate_user_input(self, distribution: AbstractQADistribution) -> str:
        page_contents = []
        for node in distribution.nodes:
            if node.type == "chunk":
                page_contents.append(node.get_property("page_content"))
        common_theme = distribution.theme
        source_text = "\n\n".join(page_contents[:4])

        question = await self.generate_question_prompt.generate(
            data=ThemeAndContext(
                theme=common_theme,
                context=source_text,
            ),
            llm=self.llm,
        )
        return question.text

    async def generate_reference(self, question: str, chunks: t.List[Node]) -> str:
        return ""

    async def critic_user_input(self, question: str) -> bool:
        critic = await self.critic_question_prompt.generate(
            data=StringIO(text=question), llm=self.llm
        )
        return critic.independence > 1 and critic.clear_intent > 1

    async def modify_user_input(
        self, question: str, distribution: AbstractQADistribution
    ) -> str:
        prompt = extend_modify_input_prompt(
            question_modification_prompt=self.question_modification_prompt,
            style=distribution.style,
            length=distribution.length,
        )
        modified_question = await prompt.generate(
            data=QuestionWithStyleAndLength(
                question=question, style=distribution.style, length=distribution.length
            ),
            llm=self.llm,
        )
        return modified_question.text
