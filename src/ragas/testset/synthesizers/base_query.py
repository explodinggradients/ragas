from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from ragas.prompt import StringIO

from .base import BaseSynthesizer, Scenario
from .prompts import (
    CriticUserInput,
    GenerateReference,
    ModifyUserInput,
    PydanticPrompt,
    QueryAndContext,
    QueryWithStyleAndLength,
    extend_modify_input_prompt,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


@dataclass
class QuerySynthesizer(BaseSynthesizer[Scenario]):
    """
    Synthesizes Question-Answer pairs. Used as a base class for other query synthesizers.

    Attributes
    ----------
    critic_query_prompt : PydanticPrompt
        The prompt used for criticizing the query.
    query_modification_prompt : PydanticPrompt
        The prompt used for modifying the query.
    generate_reference_prompt : PydanticPrompt
        The prompt used for generating the reference.
    """

    critic_query_prompt: PydanticPrompt = field(default_factory=CriticUserInput)
    query_modification_prompt: PydanticPrompt = field(default_factory=ModifyUserInput)
    generate_reference_prompt: PydanticPrompt = field(default_factory=GenerateReference)

    async def critic_query(
        self, query: str, callbacks: t.Optional[Callbacks] = None
    ) -> bool:
        callbacks = callbacks or []
        critic = await self.critic_query_prompt.generate(
            data=StringIO(text=query), llm=self.llm, callbacks=callbacks
        )
        return critic.independence > 1 and critic.clear_intent > 1

    async def modify_query(
        self, query: str, scenario: Scenario, callbacks: Callbacks
    ) -> str:
        prompt = extend_modify_input_prompt(
            query_modification_prompt=self.query_modification_prompt,
            style=scenario.style,
            length=scenario.length,
        )
        modified_query = await prompt.generate(
            data=QueryWithStyleAndLength(
                query=query,
                style=scenario.style,
                length=scenario.length,
            ),
            llm=self.llm,
            callbacks=callbacks,
        )
        return modified_query.text

    async def generate_reference(
        self,
        question: str,
        scenario: Scenario,
        callbacks: t.Optional[Callbacks] = None,
        reference_property_name: str = "page_content",
    ) -> str:
        callbacks = callbacks or []
        reference = await self.generate_reference_prompt.generate(
            data=QueryAndContext(
                query=question,
                context=self.make_reference_contexts(scenario, reference_property_name),
            ),
            llm=self.llm,
            callbacks=callbacks,
        )
        return reference.text

    @staticmethod
    def make_reference_contexts(
        scenario: Scenario, property_name: str = "page_content"
    ) -> str:
        page_contents = []
        for node in scenario.nodes:
            page_contents.append(node.get_property(property_name))
        return "\n\n".join(page_contents)
