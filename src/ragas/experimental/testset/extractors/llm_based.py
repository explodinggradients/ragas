import typing as t
from dataclasses import dataclass

from langchain_core.pydantic_v1 import BaseModel

from ragas.experimental.prompt import PydanticPrompt, StringIO
from ragas.experimental.testset.extractors.base import LLMBasedExtractor
from ragas.experimental.testset.graph import Node
from ragas.llms.base import BaseRagasLLM


# define prompts
class SummaryExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    instruction: str = "Summarize the given text in less than 10 sentences."
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[StringIO] = StringIO
    examples: t.List[t.Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text="Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations."
            ),
            StringIO(
                text="AI is revolutionizing industries by automating tasks, analyzing data, and driving innovations like self-driving cars and personalized recommendations."
            ),
        )
    ]


class Keyphrases(BaseModel):
    keyphrases: t.List[str]


class KeyphrasesExtractorPrompt(PydanticPrompt[StringIO, Keyphrases]):
    instruction: str = "Extract top 5 keyphrases from the given text."
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[Keyphrases] = Keyphrases
    examples: t.List[t.Tuple[StringIO, Keyphrases]] = [
        (
            StringIO(
                text="Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations."
            ),
            Keyphrases(
                keyphrases=[
                    "Artificial intelligence",
                    "automating tasks",
                    "healthcare",
                    "self-driving cars",
                    "personalized recommendations",
                ]
            ),
        )
    ]


class TitleExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    instruction: str = "Extract the title of the given document."
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[StringIO] = StringIO
    examples: t.List[t.Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text="Deep Learning for Natural Language Processing\n\nAbstract\n\nDeep learning has revolutionized the field of natural language processing (NLP). This paper explores various deep learning models and their applications in NLP tasks such as language translation, sentiment analysis, and text generation. We discuss the advantages and limitations of different models, and provide a comprehensive overview of the current state of the art in NLP."
            ),
            StringIO(text="Deep Learning for Natural Language Processing"),
        )
    ]


@dataclass
class SummaryExtractor(LLMBasedExtractor):
    property_name: str = "summary"
    prompt: SummaryExtractorPrompt = SummaryExtractorPrompt()

    async def _extract(self, node: Node) -> t.Any:
        node_text = node.get_property("page_content")
        if node_text is None:
            return None
        result = await self.prompt.generate(self.llm, data=StringIO(text=node_text))
        return self.property_name, result.text


@dataclass
class KeyphrasesExtractor(LLMBasedExtractor):
    property_name: str = "keyphrases"
    prompt: KeyphrasesExtractorPrompt = KeyphrasesExtractorPrompt()

    async def _extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        result = await self.prompt.generate(self.llm, data=StringIO(text=node_text))
        return self.property_name, result.keyphrases


@dataclass
class TitleExtractor(LLMBasedExtractor):
    property_name: str = "title"
    prompt: TitleExtractorPrompt = TitleExtractorPrompt()

    async def _extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        result = await self.prompt.generate(self.llm, data=StringIO(text=node_text))
        return self.property_name, result.text
