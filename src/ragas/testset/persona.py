import logging
import random
import typing as t
from dataclasses import dataclass, field

from pydantic import BaseModel

from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.llms.base import BaseRagasLLM

logger = logging.getLogger(__name__)


def default_filter(node: Node) -> bool:

    if node.type.name == "DOCUMENT" and node.properties.get("summary") is not None:
        return True
    else:
        return random.random() < 0.1


class NodeSummaries(BaseModel):
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
class PersonaGenerationPrompt(PydanticPrompt[NodeSummaries, PersonasList]):
    instruction: str = (
        "Using the provided node summaries, generate a list of possible personas who might "
        "interact with this document set for information. For each persona, include only a unique name "
        "and a brief role description summarizing who they are and their position or function."
    )
    input_model: t.Type[NodeSummaries] = NodeSummaries
    output_model: t.Type[PersonasList] = PersonasList
    examples: t.List[t.Tuple[NodeSummaries, PersonasList]] = [
        (
            NodeSummaries(
                summaries=(
                    [
                        "The Ally Lab focuses on understanding allyship, which involves actively supporting "
                        "marginalized groups to remove barriers in the workplace. Being an ally requires self-education, "
                        "empathy, active listening, humility, and courage. Allies should recognize their privilege and "
                        "take action to promote inclusivity.",
                        "The Neurodiversity in the Workplace Short Course highlights the importance of understanding "
                        "neurodiversity (including autism, ADHD, and dyslexia) to create an inclusive work environment. "
                        "The course discusses personalized communication, management styles, and reasonable accommodations.",
                        "Remote Work Challenges and Solutions discusses unique issues like communication barriers and "
                        "feelings of isolation. It recommends inclusive communication and virtual team-building activities "
                        "to support remote team members, including those from marginalized and neurodiverse backgrounds.",
                    ]
                )
            ),
            PersonasList(
                personas=[
                    Persona(
                        name="Diversity and Inclusion Officer",
                        role_description="Oversees initiatives to promote inclusivity and support for marginalized groups within the organization.",
                    ),
                    Persona(
                        name="HR Manager",
                        role_description="Manages employee support, training, and accommodations for diverse needs within the company.",
                    ),
                    Persona(
                        name="Remote Team Lead",
                        role_description="Leads a team of remote employees, focusing on inclusive communication and collaboration strategies.",
                    ),
                    Persona(
                        name="Employee Ally",
                        role_description="A team member interested in developing allyship skills to support marginalized colleagues.",
                    ),
                    Persona(
                        name="Neurodivergent Employee Advocate",
                        role_description="Works to ensure understanding and accommodations for neurodivergent employees in the workplace.",
                    ),
                ]
            ),
        )
    ]


@dataclass
class PersonaGenerator:

    llm: BaseRagasLLM
    prompt: PydanticPrompt = PersonaGenerationPrompt()
    filter_nodes: t.Callable[[Node], bool] = field(
        default_factory=lambda: default_filter
    )
    max_tokens: int = 4000

    async def generate_from_kg(self, kg: KnowledgeGraph) -> PersonasList:

        texts = []
        nodes = [node for node in kg.nodes if self.filter_nodes(node)]
        for node in nodes:
            text = node.properties.get("summary") or node.properties.get(
                "topic_description"
            ) or node.properties.get("page_content")
            if text is None:
                logger.warning(
                    f"Node {node} does not have a summary or topic description."
                )
            texts.append(text)

        random.shuffle(texts)
        prompt_input = NodeSummaries(summaries=texts[: self.max_tokens])
        response = await self.prompt.generate(data=prompt_input, llm=self.llm)
        return response
