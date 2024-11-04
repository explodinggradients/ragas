import logging
import random
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.llms.base import BaseRagasLLM
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node

logger = logging.getLogger(__name__)


def default_filter(node: Node) -> bool:

    if (
        node.type.name == "DOCUMENT"
        and node.properties.get("summary_embedding") is not None
    ):
        return True
    elif (
        node.type.name == "CHUNK"
        and node.properties.get("topic_description_embedding") is not None
    ):
        return random.random() < 0.1
    else:
        return False


class SummaryInput(BaseModel):
    summaries: t.List[str]


class Persona(BaseModel):
    name: str
    role_description: str


class PersonasList(BaseModel):
    personas: t.List[Persona]

    def __getitem__(self, key: str) -> Persona:
        for persona in self.personas:
            if persona.name == key:
                return persona
        raise KeyError(f"No persona found with name '{key}'")


# Define the prompt class
class PersonaGenerationPrompt(PydanticPrompt[SummaryInput, PersonasList]):
    instruction: str = (
        "Using the provided summaries, generate one persona for each summary who might "
        "interact with the content. For each persona, include a unique name "
        "and a brief role description of who they are."
    )
    input_model: t.Type[SummaryInput] = SummaryInput
    output_model: t.Type[PersonasList] = PersonasList
    examples: t.List[t.Tuple[SummaryInput, PersonasList]] = [
        (
            SummaryInput(
                summaries=[
                    "Guide to Digital Marketing explains strategies for engaging audiences across various online platforms.",
                    "Data Privacy Essentials discusses principles for safeguarding user data and complying with privacy regulations.",
                    "Introduction to Project Management covers key methods for planning, executing, and monitoring projects.",
                ]
            ),
            PersonasList(
                personas=[
                    Persona(
                        name="Digital Marketing Specialist",
                        role_description="Focuses on engaging audiences and growing the brand online.",
                    ),
                    Persona(
                        name="Data Privacy Officer",
                        role_description="Ensures the organization's compliance with data protection laws.",
                    ),
                    Persona(
                        name="Project Manager",
                        role_description="Oversees project timelines and ensures tasks are completed efficiently.",
                    ),
                ]
            ),
        )
    ]


@dataclass
class PersonaGenerator:

    llm: BaseRagasLLM
    num_personas: int = 5
    prompt: PydanticPrompt = PersonaGenerationPrompt()
    filter_nodes: t.Callable[[Node], bool] = field(
        default_factory=lambda: default_filter
    )

    def __post_init__(self):

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import pairwise_distances
        except ImportError:
            raise ImportError(
                "PersonaGenerator requires the 'scikit-learn' package to be installed. "
                "You can install it with 'pip install scikit-learn'."
            )

        self.pairwise_distances = pairwise_distances
        self.kmeans = KMeans(n_clusters=self.num_personas, random_state=42)

    async def generate_from_kg(self, kg: KnowledgeGraph) -> PersonasList:

        nodes = [node for node in kg.nodes if self.filter_nodes(node)]
        summaries = [
            node.properties.get("summary") or node.properties.get("topic_description")
            for node in nodes
        ]
        embeddings = []
        for node in nodes:
            embeddings.append(
                node.properties.get("summary_embedding")
                or node.properties.get("topic_description_embedding")
            )

        embeddings = np.array(embeddings)
        self.kmeans.fit(embeddings)
        labels = self.kmeans.labels_
        if labels is None:
            raise ValueError("No labels found from clustering")
        cluster_centers = self.kmeans.cluster_centers_
        top_summaries = []
        for i in range(self.num_personas):
            cluster_indices = [j for j, label in enumerate(labels) if label == i]
            _ = [summaries[j] for j in cluster_indices]
            centroid = cluster_centers[i]
            X_cluster = embeddings[cluster_indices]
            distances = self.pairwise_distances(
                X_cluster, centroid.reshape(1, -1), metric="euclidean"
            ).flatten()

            closest_index = distances.argmin()
            representative_summary = summaries[cluster_indices[closest_index]]
            top_summaries.append(representative_summary)

        prompt_input = SummaryInput(summaries=top_summaries)
        response = await self.prompt.generate(data=prompt_input, llm=self.llm)
        return response
