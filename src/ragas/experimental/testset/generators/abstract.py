import logging
import math
import random
import typing as t
from dataclasses import dataclass

from pydantic import BaseModel

from ragas.async_utils import run_async
from ragas.experimental.prompt import PydanticPrompt
from ragas.experimental.testset.generators.base import (
    BaseTestsetGenerator,
    BasicDistribution,
    QuestionLength,
    QuestionStyle,
)
from ragas.experimental.testset.graph import KnowledgeGraph, Node

logger = logging.getLogger(__name__)


class Summaries(BaseModel):
    summaries: t.List[str]
    num_themes: int


class Theme(BaseModel):
    theme: str
    description: str


class Themes(BaseModel):
    themes: t.List[Theme]


class CommonThemeFromSummaries(PydanticPrompt[Summaries, Themes]):
    input_model = Summaries
    output_model = Themes
    instruction = "Analyze the following summaries and identify given number of common themes. The themes should be concise, descriptive, and highlight a key aspect shared across the summaries."
    examples = [
        (
            Summaries(
                summaries=[
                    "Advances in artificial intelligence have revolutionized many industries. From healthcare to finance, AI algorithms are making processes more efficient and accurate. Machine learning models are being used to predict diseases, optimize investment strategies, and even recommend personalized content to users. The integration of AI into daily operations is becoming increasingly indispensable for modern businesses.",
                    "The healthcare industry is witnessing a significant transformation due to AI advancements. AI-powered diagnostic tools are improving the accuracy of medical diagnoses, reducing human error, and enabling early detection of diseases. Additionally, AI is streamlining administrative tasks, allowing healthcare professionals to focus more on patient care. Personalized treatment plans driven by AI analytics are enhancing patient outcomes.",
                    "Financial technology, or fintech, has seen a surge in AI applications. Algorithms for fraud detection, risk management, and automated trading are some of the key innovations in this sector. AI-driven analytics are helping companies to understand market trends better and make informed decisions. The use of AI in fintech is not only enhancing security but also increasing efficiency and profitability.",
                ],
                num_themes=2,
            ),
            Themes(
                themes=[
                    Theme(
                        theme="AI enhances efficiency and accuracy in various industries",
                        description="AI algorithms are improving processes across healthcare, finance, and more by increasing efficiency and accuracy.",
                    ),
                    Theme(
                        theme="AI-powered tools improve decision-making and outcomes",
                        description="AI applications in diagnostic tools, personalized treatment plans, and fintech analytics are enhancing decision-making and outcomes.",
                    ),
                ]
            ),
        )
    ]

    def process_output(self, output: Themes, input: Summaries) -> Themes:
        if len(output.themes) < input.num_themes:
            # fill the rest with empty strings
            output.themes.extend(
                [Theme(theme="none", description="")]
                * (input.num_themes - len(output.themes))
            )
        return output


class AbstractQADistribution(BasicDistribution):
    theme: str


@dataclass
class AbstractGenerator(BaseTestsetGenerator):
    def __post_init__(self):
        self.common_theme_prompt = CommonThemeFromSummaries()

    async def generate_distributions(
        self, n: int, knowledge_graph: KnowledgeGraph
    ) -> t.List[BasicDistribution]:
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

        themes: t.List[Themes] = run_async(
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
            list(QuestionStyle), k=num_clusters * num_themes
        )
        question_lengths = random.choices(
            list(QuestionLength), k=num_clusters * num_themes
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
            print(len(distributions))
        return distributions

    async def generate_question(self, distribution: BasicDistribution) -> str:
        return ""

    async def generate_answer(self, question: str, chunks: t.List[Node]) -> str:
        return ""

    async def critic_question(self, question: str) -> bool:
        return True
