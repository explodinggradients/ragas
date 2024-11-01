import typing as t
from dataclasses import dataclass

from pydantic import BaseModel

from ragas.prompt import PydanticPrompt, StringIO
from ragas.testset.graph import Node
from ragas.testset.synthesizers.prompts import ThemesAndConceptsExtractorPrompt
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


class TopicDescription(BaseModel):
    description: str


class TopicDescriptionPrompt(PydanticPrompt[StringIO, TopicDescription]):
    instruction: str = (
        "Provide a concise description of the main topic(s) discussed in the following text."
    )
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[TopicDescription] = TopicDescription
    examples: t.List[t.Tuple[StringIO, TopicDescription]] = [
        (
            StringIO(
                text="Quantum Computing\n\nQuantum computing leverages the principles of quantum mechanics to perform complex computations more efficiently than classical computers. It has the potential to revolutionize fields like cryptography, material science, and optimization problems by solving tasks that are currently intractable for classical systems."
            ),
            TopicDescription(
                description="An introduction to quantum computing and its potential to outperform classical computers in complex computations, impacting areas such as cryptography and material science."
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
    headlines: t.List[str]


class HeadlinesExtractorPrompt(PydanticPrompt[StringIO, Headlines]):
    instruction: str = (
        "Extract only the level 2 headings from the given text if they are present."
    )
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[Headlines] = Headlines
    examples: t.List[t.Tuple[StringIO, Headlines]] = [
        (
            StringIO(
                text="""\
        Section 1: Introduction
        Introduction to the topic...

        1. Main Concepts
        1.1 Key Definitions
        Explanation of core definitions...

        2. Advanced Topics
        2.1 Specialized Techniques
        Detail on various advanced techniques...

        2.2 Emerging Trends
        Description of current and emerging trends...

        3. Summary and Conclusion
        Final remarks and summary.
        """,
            ),
            Headlines(headlines=["2.1 Specialized Techniques", "2.2 Emerging Trends"]),
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
class TopicDescriptionExtractor(LLMBasedExtractor):
    """
    Extracts a concise description of the main topic(s) discussed in the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : TopicDescriptionPrompt
        The prompt used for extraction.
    """

    property_name: str = "topic_description"
    prompt: TopicDescriptionPrompt = TopicDescriptionPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        result = await self.prompt.generate(self.llm, data=StringIO(text=node_text))
        return self.property_name, result.description


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


@dataclass
class ThemesExtractor(LLMBasedExtractor):
    """
    Extracts themes from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract. Defaults to "themes".
    prompt : ThemesExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "themes"
    prompt: ThemesAndConceptsExtractorPrompt = ThemesAndConceptsExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.List[str]]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, []
        result = await self.prompt.generate(self.llm, data=StringIO(text=node_text))
        return self.property_name, result.output
