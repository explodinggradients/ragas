import typing as t

from pydantic import BaseModel

from ragas.experimental.prompt import PydanticPrompt, StringIO
from ragas.experimental.testset.generators.base import UserInputLength, UserInputStyle


class Summaries(BaseModel):
    summaries: t.List[str]
    num_themes: int


class Theme(BaseModel):
    theme: str
    description: str


class Themes(BaseModel):
    themes: t.List[Theme]


class CommonThemeFromSummaries(PydanticPrompt[Summaries, Themes]):
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


class AbstractQuestionFromTheme(PydanticPrompt[ThemeAndContext, StringIO]):
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


class UserInputWithStyleAndLength(BaseModel):
    user_input: str
    style: UserInputStyle
    length: UserInputLength


EXAMPLES_FOR_USER_INPUT_MODIFICATION = [
    # Short Length Examples
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.MISSPELLED,
            length=UserInputLength.SHORT,
        ),
        StringIO(text="How do enrgy storag solutions compare on efficincy?"),
    ),
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.PERFECT_GRAMMAR,
            length=UserInputLength.SHORT,
        ),
        StringIO(text="How do energy storage solutions compare?"),
    ),
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.POOR_GRAMMAR,
            length=UserInputLength.SHORT,
        ),
        StringIO(text="How do storag solutions compare on efficiency?"),
    ),
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.WEB_SEARCH_LIKE,
            length=UserInputLength.SHORT,
        ),
        StringIO(
            text="compare energy storage solutions efficiency cost sustainability"
        ),
    ),
    # Medium Length Examples
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.MISSPELLED,
            length=UserInputLength.MEDIUM,
        ),
        StringIO(text="How do enrgy storag solutions compare on efficincy n cost?"),
    ),
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.PERFECT_GRAMMAR,
            length=UserInputLength.MEDIUM,
        ),
        StringIO(
            text="How do energy storage solutions compare in efficiency and cost?"
        ),
    ),
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.POOR_GRAMMAR,
            length=UserInputLength.MEDIUM,
        ),
        StringIO(text="How energy storag solutions compare on efficiency and cost?"),
    ),
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.WEB_SEARCH_LIKE,
            length=UserInputLength.MEDIUM,
        ),
        StringIO(
            text="comparison of energy storage solutions efficiency cost sustainability"
        ),
    ),
    # Long Length Examples
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.MISSPELLED,
            length=UserInputLength.LONG,
        ),
        StringIO(
            text="How do various enrgy storag solutions compare in terms of efficincy, cost, and sustanbility in rnewable energy systems?"
        ),
    ),
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.PERFECT_GRAMMAR,
            length=UserInputLength.LONG,
        ),
        StringIO(
            text="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?"
        ),
    ),
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.POOR_GRAMMAR,
            length=UserInputLength.LONG,
        ),
        StringIO(
            text="How various energy storag solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?"
        ),
    ),
    (
        UserInputWithStyleAndLength(
            user_input="How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
            style=UserInputStyle.WEB_SEARCH_LIKE,
            length=UserInputLength.LONG,
        ),
        StringIO(
            text="How do various energy storage solutions compare efficiency cost sustainability renewable energy systems?"
        ),
    ),
]


class ModifyUserInput(PydanticPrompt[UserInputWithStyleAndLength, StringIO]):
    input_model = UserInputWithStyleAndLength
    output_model = StringIO
    instruction = "Modify the given question in order to fit the given style and length"
    examples: t.List[t.Tuple[UserInputWithStyleAndLength, StringIO]] = []


def extend_modify_input_prompt(
    question_modification_prompt: PydanticPrompt,
    style: UserInputStyle,
    length: UserInputLength,
) -> PydanticPrompt:
    examples = [
        example
        for example in EXAMPLES_FOR_USER_INPUT_MODIFICATION
        if example[0].style == style and example[0].length == length
    ]
    if not examples:
        raise ValueError(f"No examples found for style {style} and length {length}")
    question_modification_prompt.examples = examples
    question_modification_prompt.examples = examples
    return question_modification_prompt


class UserInputAndContext(BaseModel):
    user_input: str
    context: str


class GenerateReference(PydanticPrompt[UserInputAndContext, StringIO]):
    input_model = UserInputAndContext
    output_model = StringIO
    instruction = "Answer the following question based on the information provided in the given text."
    examples = [
        (
            UserInputAndContext(
                user_input="How does AI enhance efficiency and accuracy across different industries?",
                context="Advances in artificial intelligence have revolutionized many industries. From healthcare to finance, AI algorithms are making processes more efficient and accurate. Machine learning models are being used to predict diseases, optimize investment strategies, and even recommend personalized content to users. The integration of AI into daily operations is becoming increasingly indispensable for modern businesses.",
            ),
            StringIO(
                text="AI improves efficiency and accuracy across different industries by making processes more efficient and accurate."
            ),
        )
    ]
