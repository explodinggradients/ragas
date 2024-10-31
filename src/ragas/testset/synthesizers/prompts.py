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


class ThemesAndConcepts(BaseModel):
    output: t.List[str]


class ThemesAndConceptsExtractorPrompt(PydanticPrompt[StringIO, ThemesAndConcepts]):
    instruction: str = "Extract the main themes and concepts from the given text."
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[ThemesAndConcepts] = ThemesAndConcepts
    examples: t.List[t.Tuple[StringIO, ThemesAndConcepts]] = [
        (
            StringIO(
                text="Artificial intelligence is transforming industries by automating tasks requiring human intelligence. AI analyzes vast data quickly and accurately, driving innovations like self-driving cars and personalized recommendations."
            ),
            ThemesAndConcepts(
                output=[
                    "Artificial intelligence",
                    "Automation",
                    "Data analysis",
                    "Innovation",
                    "Self-driving cars",
                    "Personalized recommendations",
                ]
            ),
        )
    ]


    
                   

class ConceptsList(BaseModel):
    lists_of_concepts: t.List[t.List[str]]  # A list containing lists of concepts from each node

class ConceptCombinations(BaseModel):
    combinations: t.List[t.List[str]]  # Each combination is a list of concepts from different nodes

# Define the prompt class
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
    input_model: t.Type[ConceptsList] = ConceptsList  # Contains lists of concepts from each node
    output_model: t.Type[ConceptCombinations] = ConceptCombinations  # Contains list of concept combinations
    examples: t.List[t.Tuple[ConceptsList, ConceptCombinations]] = [
        (
            ConceptsList(
                lists_of_concepts=[
                    ["Artificial intelligence", "Automation"],  # Concepts from Node 1
                    ["Healthcare", "Data privacy"]             # Concepts from Node 2
                ]
            ),
            ConceptCombinations(
                combinations=[
                    ["Artificial intelligence", "Healthcare"],
                    ["Automation", "Data privacy"]
                ]
            )
        )]
    
    

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



# Define the input models
class ThemesList(BaseModel):
    themes: t.List[str]
    

# Define the output model
class PersonaThemesMapping(BaseModel):
    mapping: t.Dict[str, t.List[str]]  # Mapping from persona name to list of relevant themes

# Define the prompt class
class ThemesPersonasMatchingPrompt(PydanticPrompt[t.Tuple[ThemesList, PersonasList], PersonaThemesMapping]):
    instruction: str = (
        "Given the list of themes and the list of personas with their role descriptions, "
        "match each persona with the themes that are most relevant to them based on their role descriptions. "
        "Provide a mapping where each persona's name is associated with a list of relevant themes."
    )
    input_model: t.Type[t.Tuple[ThemesList, PersonasList]] = t.Tuple[ThemesList, PersonasList]
    output_model: t.Type[PersonaThemesMapping] = PersonaThemesMapping
    examples: t.List[t.Tuple[t.Tuple[ThemesList, PersonasList], PersonaThemesMapping]] = [
        (
            (
                ThemesList(
                    themes=[
                        "Active listening",
                        "Personalized communication",
                        "Empathy",
                        "Communication barriers",
                        "Self-education",
                        "Understanding cognitive differences",
                        "Inclusivity",
                        "Managing remote teams"
                    ]
                ),
                PersonasList(
                    personas=[
                        Persona(
                            name="HR Manager",
                            role_description="Manages employee support and training within the company."
                        ),
                        Persona(
                            name="Remote Team Lead",
                            role_description="Leads a team of remote employees, focusing on inclusive communication."
                        ),
                        Persona(
                            name="Employee Ally",
                            role_description="A team member interested in developing allyship skills."
                        ),
                    ]
                )
            ),
            PersonaThemesMapping(
                mapping={
                    "HR Manager": ["Active listening", "Personalized communication", "Self-education", "Understanding cognitive differences", "Inclusivity"],
                    "Remote Team Lead": ["Communication barriers", "Empathy", "Managing remote teams", "Inclusivity", "Active listening"],
                    "Employee Ally": ["Self-education", "Empathy", "Active listening", "Inclusivity"]
                }
            )
        )
    ]