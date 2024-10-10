import typing as t

from pydantic import BaseModel

from ragas.prompt import PydanticPrompt, StringIO
from ragas.testset.synthesizers.base import QueryLength, QueryStyle


class Summaries(BaseModel):
    summaries: t.List[str]
    num_themes: int


class Theme(BaseModel):
    theme: str
    description: str


class Themes(BaseModel):
    themes: t.List[Theme]


class CommonThemeFromSummariesPrompt(PydanticPrompt[Summaries, Themes]):
    input_model = Summaries
    output_model = Themes
    instruction = "Analyze the following summaries and identify given number of common themes. The themes should be concise, descriptive, and highlight a key aspect shared across the summaries."
    examples = [
        (
            Summaries(
                summaries=[
                    "Advances in artificial intelligence have revolutionized many industries. From healthcare to finance, AI algorithms are making processes more efficient and accurate. Machine learning models are being used to predict diseases, optimize investment strategies, and even recommend personalized content to users. The integration of AI into daily operations is becoming increasingly indispensable for modern businesses.",
                    "The healthcare industry is witnessing a significant transformation due to AI advancements. AI-powered diagnostic tools are improving the accuracy of medical diagnoses, reducing human error, and enabling early detection of diseases. Additionally, AI is streamlining administrative tasks, allowing healthcare professionals to focus more on patient care. Personalized treatment plans driven by AI analytics are enhancing patient outcomes.",
                    "Financial technology, or fintech, has seen a surge in AI applications. Algorithms for fraud detection, risk management, and automated trading are some of the key innovations in this sector. AI-driven analytics are helping companies to understand market trends better and make informed decisions. The use of AI in fintech is not only enhancing security but also increasing efficiency and profitability.",
                ],
                num_themes=2,
            ),
            Themes(
                themes=[
                    Theme(
                        theme="AI enhances efficiency and accuracy in various industries",
                        description="AI algorithms are improving processes across healthcare, finance, and more by increasing efficiency and accuracy.",
                    ),
                    Theme(
                        theme="AI-powered tools improve decision-making and outcomes",
                        description="AI applications in diagnostic tools, personalized treatment plans, and fintech analytics are enhancing decision-making and outcomes.",
                    ),
                ]
            ),
        )
    ]

    def process_output(self, output: Themes, input: Summaries) -> Themes:
        if len(output.themes) < input.num_themes:
            # fill the rest with empty strings
            output.themes.extend(
                [Theme(theme="none", description="")]
                * (input.num_themes - len(output.themes))
            )
        return output


class ThemeAndContext(BaseModel):
    theme: str
    context: str


class AbstractQueryFromTheme(PydanticPrompt[ThemeAndContext, StringIO]):
    input_model = ThemeAndContext
    output_model = StringIO
    instruction = "Generate an abstract conceptual question using the given theme that can be answered from the information in the provided context."
    examples = [
        (
            ThemeAndContext(
                theme="AI enhances efficiency and accuracy in various industries",
                context="AI is transforming various industries by improving efficiency and accuracy. For instance, in manufacturing, AI-powered robots automate repetitive tasks with high precision, reducing errors and increasing productivity. In healthcare, AI algorithms analyze medical images and patient data to provide accurate diagnoses and personalized treatment plans. Financial services leverage AI for fraud detection and risk management, ensuring quicker and more reliable decision-making. Overall, AI's ability to process vast amounts of data and learn from it enables industries to optimize operations, reduce costs, and deliver better products and services.",
            ),
            StringIO(
                text="How does AI enhance efficiency and accuracy in various industries?"
            ),
        )
    ]


class Feedback(BaseModel):
    independence: int
    clear_intent: int


class CriticUserInput(PydanticPrompt[StringIO, Feedback]):
    input_model = StringIO
    output_model = Feedback
    instruction = "Critique the synthetically generated question based on the following rubrics. Provide a score for each rubric: Independence and Clear Intent. Scores are given as low (0), medium (1), or high (2)."
    examples = [
        (
            StringIO(
                text="How does AI enhance efficiency and accuracy in various industries?"
            ),
            Feedback(independence=2, clear_intent=2),
        ),
        (
            StringIO(text="Explain the benefits of AI."),
            Feedback(independence=1, clear_intent=1),
        ),
        (
            StringIO(text="How does AI?"),
            Feedback(independence=0, clear_intent=0),
        ),
    ]


class QueryWithStyleAndLength(BaseModel):
    query: str
    style: QueryStyle
    length: QueryLength


EXAMPLES_FOR_USER_INPUT_MODIFICATION = [
    # Short Length Examples
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.MISSPELLED,
            length=QueryLength.SHORT,
        ),
        StringIO(text="How do enrgy storag solutions compare on efficincy?"),
    ),
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.PERFECT_GRAMMAR,
            length=QueryLength.SHORT,
        ),
        StringIO(text="How do energy storage solutions compare?"),
    ),
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.POOR_GRAMMAR,
            length=QueryLength.SHORT,
        ),
        StringIO(text="How do storag solutions compare on efficiency?"),
    ),
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.WEB_SEARCH_LIKE,
            length=QueryLength.SHORT,
        ),
        StringIO(
            text="compare energy storage solutions efficiency cost sustainability"
        ),
    ),
    # Medium Length Examples
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.MISSPELLED,
            length=QueryLength.MEDIUM,
        ),
        StringIO(text="How do enrgy storag solutions compare on efficincy n cost?"),
    ),
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.PERFECT_GRAMMAR,
            length=QueryLength.MEDIUM,
        ),
        StringIO(
            text="How do energy storage solutions compare in efficiency and cost?"
        ),
    ),
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.POOR_GRAMMAR,
            length=QueryLength.MEDIUM,
        ),
        StringIO(text="How energy storag solutions compare on efficiency and cost?"),
    ),
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.WEB_SEARCH_LIKE,
            length=QueryLength.MEDIUM,
        ),
        StringIO(
            text="comparison of energy storage solutions efficiency cost sustainability"
        ),
    ),
    # Long Length Examples
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.MISSPELLED,
            length=QueryLength.LONG,
        ),
        StringIO(
            text="How do various enrgy storag solutions compare in terms of efficincy, cost, and sustanbility in rnewable energy systems?"
        ),
    ),
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.PERFECT_GRAMMAR,
            length=QueryLength.LONG,
        ),
        StringIO(
            text="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?"
        ),
    ),
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.POOR_GRAMMAR,
            length=QueryLength.LONG,
        ),
        StringIO(
            text="How various energy storag solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?"
        ),
    ),
    (
        QueryWithStyleAndLength(
            query="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=QueryStyle.WEB_SEARCH_LIKE,
            length=QueryLength.LONG,
        ),
        StringIO(
            text="How do various energy storage solutions compare efficiency cost sustainability renewable energy systems?"
        ),
    ),
]


class ModifyUserInput(PydanticPrompt[QueryWithStyleAndLength, StringIO]):
    input_model = QueryWithStyleAndLength
    output_model = StringIO
    instruction = "Modify the given question in order to fit the given style and length"
    examples: t.List[t.Tuple[QueryWithStyleAndLength, StringIO]] = []


def extend_modify_input_prompt(
    query_modification_prompt: PydanticPrompt,
    style: QueryStyle,
    length: QueryLength,
) -> PydanticPrompt:
    examples = [
        example
        for example in EXAMPLES_FOR_USER_INPUT_MODIFICATION
        if example[0].style == style and example[0].length == length
    ]
    if not examples:
        raise ValueError(f"No examples found for style {style} and length {length}")
    query_modification_prompt.examples = examples
    query_modification_prompt.examples = examples
    return query_modification_prompt


class QueryAndContext(BaseModel):
    query: str
    context: str


class GenerateReference(PydanticPrompt[QueryAndContext, StringIO]):
    input_model = QueryAndContext
    output_model = StringIO
    instruction = "Answer the following question based on the information provided in the given text."
    examples = [
        (
            QueryAndContext(
                query="How does AI enhance efficiency and accuracy across different industries?",
                context="Advances in artificial intelligence have revolutionized many industries. From healthcare to finance, AI algorithms are making processes more efficient and accurate. Machine learning models are being used to predict diseases, optimize investment strategies, and even recommend personalized content to users. The integration of AI into daily operations is becoming increasingly indispensable for modern businesses.",
            ),
            StringIO(
                text="AI improves efficiency and accuracy across different industries by making processes more efficient and accurate."
            ),
        )
    ]


class KeyphrasesAndNumConcepts(BaseModel):
    keyphrases: t.List[str]
    num_concepts: int


class Concepts(BaseModel):
    concepts: t.Dict[str, t.List[str]]


class CommonConceptsFromKeyphrases(PydanticPrompt[KeyphrasesAndNumConcepts, Concepts]):
    input_model = KeyphrasesAndNumConcepts
    output_model = Concepts
    instruction = "Identify a list of common concepts from the given list of key phrases for comparing the given theme across reports."
    examples = [
        (
            KeyphrasesAndNumConcepts(
                keyphrases=[
                    "fast charging",
                    "long battery life",
                    "OLED display",
                    "waterproof",
                ],
                num_concepts=4,
            ),
            Concepts(
                concepts={
                    "Charging": [
                        "fast charging",
                        "long battery life",
                        "OLED display",
                        "waterproof",
                    ],
                    "Battery Life": [
                        "long battery life",
                        "extended battery",
                        "durable battery",
                        "prolonged battery",
                    ],
                    "Display": [
                        "OLED display",
                        "HD display",
                        "AMOLED display",
                        "retina display",
                    ],
                    "Water/Dust Resistance": [
                        "waterproof",
                        "dust resistant",
                        "splash proof",
                        "water resistant",
                    ],
                }
            ),
        )
    ]

    def process_output(
        self, output: Concepts, input: KeyphrasesAndNumConcepts
    ) -> Concepts:
        if len(output.concepts) < input.num_concepts:
            # fill the rest with empty strings
            output.concepts.update(
                {
                    "Concept" + str(i): []
                    for i in range(input.num_concepts - len(output.concepts))
                }
            )
        return output


class CAQInput(BaseModel):
    concept: str
    keyphrases: t.List[str]
    summaries: t.List[str]


class ComparativeAbstractQuery(PydanticPrompt[CAQInput, StringIO]):
    input_model = CAQInput
    output_model = StringIO
    instruction = "Generate an abstract comparative question based on the given concept, keyphrases belonging to that concept, and summaries of reports."
    examples = [
        (
            CAQInput(
                concept="Battery Life",
                keyphrases=[
                    "long battery life",
                    "extended battery",
                    "durable battery",
                    "prolonged battery",
                ],
                summaries=[
                    "The device offers a long battery life, capable of lasting up to 24 hours on a single charge.",
                    "Featuring an extended battery, the product can function for 20 hours with heavy usage.",
                    "With a durable battery, this model ensures 22 hours of operation under normal conditions.",
                    "The battery life is prolonged, allowing the gadget to be used for up to 18 hours on one charge.",
                ],
            ),
            StringIO(
                text="How do the battery life claims and performance metrics compare across different reports for devices featuring long battery life, extended battery, durable battery, and prolonged battery?"
            ),
        )
    ]


class SpecificQuestionInput(BaseModel):
    title: str
    keyphrase: str
    text: str


class SpecificQuery(PydanticPrompt[SpecificQuestionInput, StringIO]):
    input_model = SpecificQuestionInput
    output_model = StringIO
    instruction = "Given the title of a text and a text chunk, along with a keyphrase from the chunk, generate a specific question related to the keyphrase.\n\n"
    "1. Read the title and the text chunk.\n"
    "2. Identify the context of the keyphrase within the text chunk.\n"
    "3. Formulate a question that directly relates to the keyphrase and its context within the chunk.\n"
    "4. Ensure the question is clear, specific, and relevant to the keyphrase."
    examples = [
        (
            SpecificQuestionInput(
                title="The Impact of Artificial Intelligence on Modern Healthcare",
                keyphrase="personalized treatment plans",
                text="Artificial intelligence (AI) is revolutionizing healthcare by improving diagnostic accuracy and enabling personalized treatment plans. AI algorithms analyze vast amounts of medical data to identify patterns and predict patient outcomes, which enhances the decision-making process for healthcare professionals.",
            ),
            StringIO(
                text="How does AI contribute to the development of personalized treatment plans in healthcare?"
            ),
        )
    ]
