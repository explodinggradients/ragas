import typing as t

from pydantic import BaseModel

from ragas.prompt import PydanticPrompt
from ragas.testset.persona import Persona, PersonaList


class ThemesList(BaseModel):
    themes: t.List[str]


class ThemesPersonasInput(BaseModel):
    themes: ThemesList
    personas: PersonaList


# Define the output model
class PersonaThemesMapping(BaseModel):
    mapping: t.Dict[str, t.List[str]]


# Define the prompt class
class ThemesPersonasMatchingPrompt(
    PydanticPrompt[ThemesPersonasInput, PersonaThemesMapping]
):
    instruction: str = (
        "Given the list of themes and the list of personas with their role descriptions, "
        "match each persona with the themes that are most relevant to them based on their role descriptions. "
        "Provide a mapping where each persona's name is associated with a list of relevant themes."
    )
    input_model: t.Type[ThemesPersonasInput] = ThemesPersonasInput
    output_model: t.Type[PersonaThemesMapping] = PersonaThemesMapping
    examples: t.List[t.Tuple[ThemesPersonasInput, PersonaThemesMapping]] = [
        (
            ThemesPersonasInput(
                themes=ThemesList(
                    themes=[
                        "Active listening",
                        "Personalized communication",
                        "Empathy",
                        "Communication barriers",
                        "Self-education",
                        "Understanding cognitive differences",
                        "Inclusivity",
                        "Managing remote teams",
                    ]
                ),
                personas=PersonaList(
                    personas=[
                        Persona(
                            name="HR Manager",
                            role_description="Manages employee support and training within the company.",
                        ),
                        Persona(
                            name="Remote Team Lead",
                            role_description="Leads a team of remote employees, focusing on inclusive communication.",
                        ),
                        Persona(
                            name="Employee Ally",
                            role_description="A team member interested in developing allyship skills.",
                        ),
                    ]
                ),
            ),
            PersonaThemesMapping(
                mapping={
                    "HR Manager": [
                        "Active listening",
                        "Personalized communication",
                        "Self-education",
                        "Understanding cognitive differences",
                        "Inclusivity",
                    ],
                    "Remote Team Lead": [
                        "Communication barriers",
                        "Empathy",
                        "Managing remote teams",
                        "Inclusivity",
                        "Active listening",
                    ],
                    "Employee Ally": [
                        "Self-education",
                        "Empathy",
                        "Active listening",
                        "Inclusivity",
                    ],
                }
            ),
        )
    ]
