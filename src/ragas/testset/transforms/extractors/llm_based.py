import typing as t
from dataclasses import dataclass

from pydantic import BaseModel

from ragas.prompt import PydanticPrompt, StringIO
from ragas.testset.graph import Node
from ragas.testset.transforms.base import LLMBasedExtractor


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


class Headlines(BaseModel):
    headlines: t.Dict[str, t.List[str]]


class HeadlinesExtractorPrompt(PydanticPrompt[StringIO, Headlines]):
    instruction: str = "Extract the headlines from the given text."
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[Headlines] = Headlines
    examples: t.List[t.Tuple[StringIO, Headlines]] = [
        (
            StringIO(
                text="""\
Some Title
1. Introduction and Related Work

1.1 Conditional Computation
Exploiting scale in both training data and model size has been central to the success of deep learning...
1.2 Our Approach: The Sparsely-Gated Mixture-of-Experts Layer
Our approach to conditional computation is to introduce a new type of general purpose neural network component...
1.3 Related Work on Mixtures of Experts
Since its introduction more than two decades ago (Jacobs et al., 1991; Jordan & Jacobs, 1994), the mixture-of-experts approach..

2. The Sparsely-Gated Mixture-of-Experts Layer
2.1 Architecture
The sparsely-gated mixture-of-experts layer is a feedforward neural network layer that consists of a number of expert networks and a single gating network...
""",
            ),
            Headlines(
                headlines={
                    "1. Introduction and Related Work": [
                        "1.1 Conditional Computation",
                        "1.2 Our Approach: The Sparsely-Gated Mixture-of-Experts Layer",
                        "1.3 Related Work on Mixtures of Experts",
                    ],
                    "2. The Sparsely-Gated Mixture-of-Experts Layer": [
                        "2.1 Architecture"
                    ],
                },
            ),
        ),
    ]


class NamedEntities(BaseModel):
    ORG: t.List[str]
    LOC: t.List[str]
    PER: t.List[str]
    MISC: t.List[str]


class NEROutput(BaseModel):
    entities: NamedEntities


class NERPrompt(PydanticPrompt[StringIO, NEROutput]):
    instruction: str = "Extract named entities from the given text."
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[NEROutput] = NEROutput
    examples: t.List[t.Tuple[StringIO, NEROutput]] = [
        (
            StringIO(
                text="Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations."
            ),
            NEROutput(
                entities=NamedEntities(
                    ORG=["Artificial intelligence"],
                    LOC=["healthcare", "finance"],
                    PER=[],
                    MISC=["self-driving cars", "personalized recommendations"],
                )
            ),
        )
    ]


@dataclass
class SummaryExtractor(LLMBasedExtractor):
    """
    Extracts a summary from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : SummaryExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "summary"
    prompt: SummaryExtractorPrompt = SummaryExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        result = await self.prompt.generate(self.llm, data=StringIO(text=node_text))
        return self.property_name, result.text


@dataclass
class KeyphrasesExtractor(LLMBasedExtractor):
    """
    Extracts top 5 keyphrases from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : KeyphrasesExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "keyphrases"
    prompt: KeyphrasesExtractorPrompt = KeyphrasesExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        result = await self.prompt.generate(self.llm, data=StringIO(text=node_text))
        return self.property_name, result.keyphrases


@dataclass
class TitleExtractor(LLMBasedExtractor):
    """
    Extracts the title from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : TitleExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "title"
    prompt: TitleExtractorPrompt = TitleExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        result = await self.prompt.generate(self.llm, data=StringIO(text=node_text))
        return self.property_name, result.text


@dataclass
class HeadlinesExtractor(LLMBasedExtractor):
    """
    Extracts the headlines from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : HeadlinesExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "headlines"
    prompt: HeadlinesExtractorPrompt = HeadlinesExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        result = await self.prompt.generate(self.llm, data=StringIO(text=node_text))
        if result is None:
            return self.property_name, None
        return self.property_name, result.headlines


@dataclass
class NERExtractor(LLMBasedExtractor):
    """
    Extracts named entities from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract. Defaults to "entities".
    prompt : NERPrompt
        The prompt used for extraction.
    """

    property_name: str = "entities"
    prompt: NERPrompt = NERPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Dict[str, t.List[str]]]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, {}
        result = await self.prompt.generate(self.llm, data=StringIO(text=node_text))
        return self.property_name, result.entities.model_dump()
