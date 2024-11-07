import typing as t

from pydantic import BaseModel, Field

from ragas.prompt import PydanticPrompt
from ragas.testset.persona import Persona


class ConceptsList(BaseModel):
    lists_of_concepts: t.List[t.List[str]] = Field(
        description="A list containing lists of concepts from each node"
    )
    max_combinations: int = Field(
        description="The maximum number of concept combinations to generate", default=5
    )


class ConceptCombinations(BaseModel):
    combinations: t.List[t.List[str]]


class ConceptCombinationPrompt(PydanticPrompt[ConceptsList, ConceptCombinations]):
    instruction: str = (
        "Form combinations by pairing concepts from at least two different lists.\n"
        "**Instructions:**\n"
        "- Review the concepts from each node.\n"
        "- Identify concepts that can logically be connected or contrasted.\n"
        "- Form combinations that involve concepts from different nodes.\n"
        "- Each combination should include at least one concept from two or more nodes.\n"
        "- List the combinations clearly and concisely.\n"
        "- Do not repeat the same combination more than once."
    )
    input_model: t.Type[ConceptsList] = (
        ConceptsList  # Contains lists of concepts from each node
    )
    output_model: t.Type[ConceptCombinations] = (
        ConceptCombinations  # Contains list of concept combinations
    )
    examples: t.List[t.Tuple[ConceptsList, ConceptCombinations]] = [
        (
            ConceptsList(
                lists_of_concepts=[
                    ["Artificial intelligence", "Automation"],  # Concepts from Node 1
                    ["Healthcare", "Data privacy"],  # Concepts from Node 2
                ],
                max_combinations=2,
            ),
            ConceptCombinations(
                combinations=[
                    ["Artificial intelligence", "Healthcare"],
                    ["Automation", "Data privacy"],
                ]
            ),
        )
    ]


class QueryConditions(BaseModel):
    persona: Persona
    themes: t.List[str]
    query_style: str
    query_length: str
    context: t.List[str]


class GeneratedQueryAnswer(BaseModel):
    query: str
    answer: str


class QueryAnswerGenerationPrompt(
    PydanticPrompt[QueryConditions, GeneratedQueryAnswer]
):
    instruction: str = (
        "Generate a query and answer based on the specified conditions (persona, themes, style, length) "
        "and the provided context. Ensure the answer is fully faithful to the context, only using information "
        "directly from the nodes provided."
        "### Instructions:\n"
        "1. **Generate a Query**: Based on the context, persona, themes, style, and length, create a question "
        "that aligns with the persona’s perspective and reflects the themes.\n"
        "2. **Generate an Answer**: Using only the content from the provided context, create a faithful and detailed  answer to "
        "the query. Do not include any information that not in or cannot be inferred from the given context.\n"
        "### Example Outputs:\n\n"
    )
    input_model: t.Type[QueryConditions] = QueryConditions
    output_model: t.Type[GeneratedQueryAnswer] = GeneratedQueryAnswer
