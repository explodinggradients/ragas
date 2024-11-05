import logging
import random
import typing as t
from dataclasses import dataclass

import numpy as np
from langchain_core.callbacks import Callbacks
from pydantic import BaseModel
from tqdm import tqdm

from ragas.llms.base import BaseRagasLLM
from ragas.prompt import PydanticPrompt, StringIO
from ragas.testset.graph import KnowledgeGraph, Node

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


def default_filter(node: Node) -> bool:

    if (
        node.type.name == "DOCUMENT"
        and node.properties.get("summary_embedding") is not None
    ):
        return True
    else:
        return False


class Persona(BaseModel):
    name: str
    role_description: str


class PersonaGenerationPrompt(PydanticPrompt[StringIO, Persona]):
    instruction: str = (
        "Using the provided summary, generate a single persona who would likely "
        "interact with or benefit from the content. Include a unique name and a "
        "concise role description of who they are."
    )
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[Persona] = Persona
    examples: t.List[t.Tuple[StringIO, Persona]] = [
        (
            StringIO(
                text="Guide to Digital Marketing explains strategies for engaging audiences across various online platforms."
            ),
            Persona(
                name="Digital Marketing Specialist",
                role_description="Focuses on engaging audiences and growing the brand online.",
            ),
        )
    ]


@dataclass
class PersonaList:
    personas: t.List[Persona]

    def __getitem__(self, key: str) -> t.Optional[Persona]:
        for persona in self.personas:
            if persona.name == key:
                return persona
        return None

    @classmethod
    async def from_kg(
        cls,
        llm: BaseRagasLLM,
        kg: KnowledgeGraph,
        persona_generation_prompt: PersonaGenerationPrompt = PersonaGenerationPrompt(),
        num_personas: int = 5,
        filter_fn: t.Callable[[Node], bool] = default_filter,
        callbacks: Callbacks = [],
    ) -> "PersonaList":

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import pairwise_distances
        except ImportError:
            raise ImportError(
                "PersonaGenerator requires the 'scikit-learn' package to be installed. "
                "You can install it with 'pip install scikit-learn'."
            )

        kmeans = KMeans(n_clusters=num_personas, random_state=42)

        nodes = [node for node in kg.nodes if filter_fn(node)]
        summaries = [node.properties.get("summary") for node in nodes]
        if len(summaries) < num_personas:
            logger.warning(
                f"Only {len(summaries)} summaries found, randomly duplicating to reach {num_personas} personas."
            )
            summaries.extend(random.choices(summaries, k=num_personas - len(summaries)))
        summaries = [summary for summary in summaries if isinstance(summary, str)]

        embeddings = []
        for node in nodes:
            embeddings.append(node.properties.get("summary_embedding"))

        embeddings = np.array(embeddings)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        if labels is None:
            raise ValueError("No labels found from clustering")
        cluster_centers = kmeans.cluster_centers_
        persona_list = []
        for i in tqdm(range(num_personas), desc="Generating personas"):
            cluster_indices = [j for j, label in enumerate(labels) if label == i]
            _ = [summaries[j] for j in cluster_indices]
            centroid = cluster_centers[i]
            X_cluster = embeddings[cluster_indices]
            distances = pairwise_distances(
                X_cluster, centroid.reshape(1, -1), metric="euclidean"
            ).flatten()

            closest_index = distances.argmin()
            representative_summary: str = summaries[cluster_indices[closest_index]]
            persona = await persona_generation_prompt.generate(
                llm=llm, data=StringIO(text=representative_summary), callbacks=callbacks
            )
            persona_list.append(persona)

        return cls(personas=persona_list)
