from dataclasses import dataclass, field

from ragas.experimental.prompt import StringIO

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


@dataclass
class QuerySynthesizer(BaseSynthesizer[Scenario]):
    critic_query_prompt: PydanticPrompt = field(default_factory=CriticUserInput)
    query_modification_prompt: PydanticPrompt = field(default_factory=ModifyUserInput)
    generate_reference_prompt: PydanticPrompt = field(default_factory=GenerateReference)

    async def critic_query(self, query: str) -> bool:
        critic = await self.critic_query_prompt.generate(
            data=StringIO(text=query), llm=self.llm
        )
        return critic.independence > 1 and critic.clear_intent > 1

    async def modify_query(self, query: str, scenario: Scenario) -> str:
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
        )
        return modified_query.text

    async def generate_reference(
        self,
        question: str,
        scenario: Scenario,
        reference_property_name: str = "page_content",
    ) -> str:
        reference = await self.generate_reference_prompt.generate(
            data=QueryAndContext(
                query=question,
                context=self.make_reference_contexts(scenario, reference_property_name),
            ),
            llm=self.llm,
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
