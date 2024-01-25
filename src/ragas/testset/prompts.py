from langchain_core.prompts import HumanMessagePromptTemplate

from ragas.llms.prompt import Prompt

seed_question_prompt = Prompt(
    name="seed_question",
    instruction="""Generate a question from given context satisfying the rules given below:
    2.The question should be framed such that it must be clearly understood without providing context.
    3.The question should be fully answerable from information present in given context.""",
    examples=[
        {
            "context": "The Eiffel Tower in Paris was originally intended as a temporary structure, built for the 1889 World's Fair. It was almost dismantled in 1909 but was saved because it was repurposed as a giant radio antenna.",
            "output": "Who built the Eiffel Tower?",
        },
    ],
    input_keys=["context"],
    output_key="output",
    output_type="string",
    language="english",
)


reasoning_question_prompt = Prompt(
    name="reasoning_question",
    instruction="""Complicate the given question by rewriting question into a multi-hop reasoning question based on the provided context.
    Answering the question should require the reader to make multiple logical connections or inferences using the information available in given context.
    Rules to follow when rewriting question:
    1. Ensure that the rewritten question can be answered entirely from the information present in the contexts.
    2. Do not frame questions that contains more than 15 words. Use abbreviation wherever possible.
    3. Make sure the question is clear and unambiguous.
    4. phrases like 'based on the provided context','according to the context',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. It has several cities, including Paris, Lyon, and Marseille. Paris is not only known for its cultural landmarks like the Eiffel Tower and the Louvre Museum but also as the administrative center.",
            "output": "Linking the Eiffel Tower and administrative center, which city stands as both?",
        },
        {
            "question": "What does the append() method do in Python?",
            "context": "In Python, lists are used to store multiple items in a single variable. Lists are one of 4 built-in data types used to store collections of data. The append() method adds a single item to the end of a list.",
            "output": "If a list represents a variable collection, what method extends it by one item?",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="string",
    language="english",
)


multi_context_question_prompt = Prompt(
    name="multi_context_question",
    instruction="""
    The task is to rewrite and complicate the given question in a way that answering it requires information derived from both context1 and context2. 
    Follow the rules given below while rewriting the question.
        1. The rewritten question should not be very long. Use abbreviation wherever possible.
        2. The rewritten question must be reasonable and must be understood and responded by humans.
        3. The rewritten question must be fully answerable from information present in context1 and context2. 
        4. Read and understand both contexts and rewrite the question so that answering requires insight from both context1 and context2.
        5. phrases like 'based on the provided context','according to the context?',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What process turns plants green?",
            "context1": "Chlorophyll is the pigment that gives plants their green color and helps them photosynthesize.",
            "context2": "Photosynthesis in plants typically occurs in the leaves where chloroplasts are concentrated.",
            "output": "In which plant structures does the pigment responsible for their verdancy facilitate energy production?",
        },
        {
            "question": "How do you calculate the area of a rectangle?",
            "context1": "The area of a shape is calculated based on the shape's dimensions. For rectangles, this involves multiplying the length and width.",
            "context2": "Rectangles have four sides with opposite sides being equal in length. They are a type of quadrilateral.",
            "output": "What multiplication involving equal opposites yields a quadrilateral's area?",
        },
    ],
    input_keys=["question", "context1", "context2"],
    output_key="output",
    output_type="string",
    language="english",
)

conditional_question_prompt = Prompt(
    name="conditional_question",
    instruction="""Rewrite the provided question to increase its complexity by introducing a conditional element.
    The goal is to make the question more intricate by incorporating a scenario or condition that affects the context of the question.
    Follow the rules given below while rewriting the question.
        1. The rewritten question should not be longer than 25 words. Use abbreviation wherever possible.
        2. The rewritten question must be reasonable and must be understood and responded by humans.
        3. The rewritten question must be fully answerable from information present context.
        4. phrases like 'provided context','according to the context?',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What is the function of the roots of a plant?",
            "context": "The roots of a plant absorb water and nutrients from the soil, anchor the plant in the ground, and store food.",
            "output": "What dual purpose do plant roots serve concerning soil nutrients and stability?",
        },
        {
            "question": "How do vaccines protect against diseases?",
            "context": "Vaccines protect against diseases by stimulating the body's immune response to produce antibodies, which recognize and combat pathogens.",
            "output": "How do vaccines utilize the body's immune system to defend against pathogens?",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="string",
    language="english",
)


compress_question_prompt = Prompt(
    name="compress_question",
    instruction="""Rewrite the following question to make it more indirect and shorter while retaining the essence of the original question.
    The goal is to create a question that conveys the same meaning but in a less direct manner. The rewritten question should shorter so use abbreviation wherever possible.""",
    examples=[
        {
            "question": "What is the distance between the Earth and the Moon?",
            "output": "How far is the Moon from Earth?",
        },
        {
            "question": "What ingredients are required to bake a chocolate cake?",
            "output": "What's needed for a chocolate cake?",
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="string",
    language="english",
)


conversational_question_prompt = Prompt(
    name="conversation_question",
    instruction="""Reformat the provided question into two separate questions as if it were to be part of a conversation. Each question should focus on a specific aspect or subtopic related to the original question.
    Follow the rules given below while rewriting the question.
        1. The rewritten question should not be longer than 25 words. Use abbreviation wherever possible.
        2. The rewritten question must be reasonable and must be understood and responded by humans.
        3. The rewritten question must be fully answerable from information present context.
        4. phrases like 'provided context','according to the context?',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What are the advantages and disadvantages of remote work?",
            "output": {
                "first_question": "What are the benefits of remote work?",
                "second_question": "On the flip side, what challenges are encountered when working remotely?",
            },
        }
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="english",
)

context_scoring_prompt = Prompt(
    name="score_context",
    instruction="""Given a context, complete the two following tasks and output answer valid json format
1.Evaluate the provided context and assign a numerical score between 0 and 10 based on the following criteria:
    - Award a high score to context that thoroughly delves into and explains concepts.
    - Assign a lower score to context that contains excessive references, acknowledgments, personal information, or other non-essential elements.""",
    examples=[
        {
            "context": "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.",
            "output": {"score": 6.0},
        }
    ],
    input_keys=["context"],
    output_key="output",
    output_type="json",
    language="english",
)

question_rewrite_prompt = Prompt(
    name="rewrite_question",
    instruction="""Given a context, transform the given question to be clear and standalone by replacing its coreferences with specific details from the context:""",
    examples=[
        {
            "context": "The Eiffel Tower was constructed using iron and was originally intended as a temporary exhibit for the 1889 World's Fair held in Paris. Despite its initial temporary purpose, the Eiffel Tower quickly became a symbol of Parisian ingenuity and an iconic landmark of the city, attracting millions of visitors each year. The tower's design, created by Gustave Eiffel, was initially met with criticism from some French artists and intellectuals, but it has since been celebrated as a masterpiece of structural engineering and architectural design.",
            "question": "Who created the design for the Tower?",
            "output": "Who created the design for the Eiffel Tower?",
        },
        {
            "context": "'Exploring Zero-Shot Learning in Neural Networks' was published by Smith and Lee in 2021, focusing on the application of zero-shot learning techniques in artificial intelligence.",
            "question": "What datasets were used for the zero-shot evaluations in this study?",
            "output": "What datasets were used for the zero-shot evaluations Exploring Zero-Shot Learning in Neural Networks paper?",
        },
    ],
    input_keys=["context", "question"],
    output_key="output",
    output_type="string",
    language="english",
)


filter_question_prompt = Prompt(
    name="filter_question",
    instruction="""Given a question, classify it based on clarity and specificity""",
    examples=[
        {
            "question": "What is the discovery about space?",
            "output": {
                "reason": "The question is too vague and does not specify which discovery about space it is referring to.",
                "verdit": "No",
            },
        },
        {
            "question": "What caused the Great Depression?",
            "output": {
                "reason": "The question is specific and refers to a well-known historical economic event, making it clear and answerable.",
                "verdict": "Yes",
            },
        },
        {
            "question": "What is the keyword that best describes the paper's focus in natural language understanding tasks?",
            "output": {
                "reason": "The question mentions a 'paper' in it without referring it's name which makes it unclear without it",
                "verdict": "No",
            },
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "output": {
                "reason": "The question is clear and refers to a specific work by name therefore it is clear",
                "verdict": "Yes",
            },
        },
        {
            "question": "What did the study mention?",
            "output": {
                "reason": "The question is vague and does not specify which study it is referring to",
                "verdict": "No",
            },
        },
        {
            "question": "What is the focus of the REPLUG paper?",
            "output": {
                "reason": "The question refers to a specific work by it's name hence can be understood",
                "verdict": "Yes",
            },
        },
        {
            "question": "What is the purpose of the reward-driven stage in the training process?",
            "output": {
                "reason": "The question lacks specific context regarding the type of training process, making it potentially ambiguous and open to multiple interpretations.",
                "verdict": "No",
            },
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="english",
)

evolution_elimination_prompt = Prompt(
    name="evolution_elimination",
    instruction="""Check if the given two questions are equal based on following requirements:
    1. They have same constraints and requirements.
    2. They have same depth and breadth of the inquiry.""",
    examples=[
        {
            "question1": "What are the primary causes of climate change?",
            "question2": "What factors contribute to global warming?",
            "output": {
                "reason": "While both questions deal with environmental issues, 'climate change' encompasses broader changes than 'global warming', leading to different depths of inquiry.",
                "verdict": "Not Equal",
            },
        },
        {
            "question1": "How does photosynthesis work in plants?",
            "question2": "Can you explain the process of photosynthesis in plants?",
            "output": {
                "reason": "Both questions ask for an explanation of the photosynthesis process in plants, sharing the same depth, breadth, and requirements for the answer.",
                "verdict": "Equal",
            },
        },
    ],
    input_keys=["question1", "question2"],
    output_key="output",
    output_type="json",
    language="english",
)

question_answer_prompt = Prompt(
    name="answer_formulate",
    instruction="""Answer the question using the information from the given context. Answer '-1' if answer is not present in the context.""",
    examples=[
        {
            "context": """The novel '1984' by George Orwell is set in a dystopian future where the world is divided into three superstates. The story follows the life of Winston Smith, who lives in Oceania, a superstate constantly at war.""",
            "question": "In which superstate does Winston Smith live in the novel '1984'?",
            "answer": "Winston Smith lives in the superstate of Oceania in the novel '1984'.",
        },
        {
            "context": """The novel "Pride and Prejudice" by Jane Austen revolves around the character Elizabeth Bennet and her family. The story is set in the 19th century in rural England and deals with issues of marriage, morality, and misconceptions.""",
            "question": "What year was 'Pride and Prejudice' published?",
            "answer": "-1",
        },
    ],
    input_keys=["context", "question"],
    output_key="answer",
    output_type="string",
    language="english",
)


## TODO: remove this

SEED_QUESTION = HumanMessagePromptTemplate.from_template(
    """
Generate two questions from given context satisfying the rules given below:
    2.The question should be framed such that it must be clearly understood without providing context.
    3.The question should be fully answerable from information present in given context.
    

{demonstration}


Context:
{context}
Questions:"""  # noqa: E501
)

TABLE_QA = HumanMessagePromptTemplate.from_template(
    """
Frame a question from the given table following the rules given below
    - Do no use phrases like 'provided context','provided table' etc in the question
    
Context:
Table 2: Local Library Statistics

Month	New Memberships	Books Loaned	eBooks Downloaded
January	150	1200	950
February	120	1100	1000
March	200	1400	1100

Framed Question from Table: How many books were loaned in January?

Context:
{context}

Framed Question from Table:"""  # noqa: E501
)


REASONING_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Complicate the given question by rewriting question into a multi-hop reasoning question based on the provided context. 
Answering the question should require the reader to make multiple logical connections or inferences using the information available in given context. 
Rules to follow when rewriting question:
1. Ensure that the rewritten question can be answered entirely from the information present in the contexts.
2. Do not frame questions that contains more than 15 words. Use abbreviation wherever possible.
3. Make sure the question is clear and unambiguous. 
4. phrases like 'based on the provided context','according to the context',etc are not allowed to appear in the question.

Initial Question:
What is the capital of France?

Given Context:
France is a country in Western Europe. It has several cities, including Paris, Lyon, and Marseille. Paris is not only known for its cultural landmarks like the Eiffel Tower and the Louvre Museum but also as the administrative center.

Complicated Multi-Hop Question:
Linking the Eiffel Tower and administrative center, which city stands as both?

Initial Question:
What does the append() method do in Python?

Given Context:
In Python, lists are used to store multiple items in a single variable. Lists are one of 4 built-in data types used to store collections of data. The append() method adds a single item to the end of a list.

Complicated Multi-Hop Question:
If a list represents a variable collection, what method extends it by one item?

Initial Question: 
{question}
Given Context:
{context}

Complicated Multi-Hop Question
"""  # noqa: E501
)

MULTICONTEXT_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
The task is to rewrite and complicate the given question in a way that answering it requires information derived from both context1 and context2. 
Follow the rules given below while rewriting the question.
    1. The rewritten question should not be very long. Use abbreviation wherever possible.
    2. The rewritten question must be reasonable and must be understood and responded by humans.
    3. The rewritten question must be fully answerable from information present in context1 and context2. 
    4. Read and understand both contexts and rewrite the question so that answering requires insight from both context1 and context2.
    5. phrases like 'based on the provided context','according to the context?',etc are not allowed to appear in the question.

Initial Question:
What process turns plants green?

Context1:
Chlorophyll is the pigment that gives plants their green color and helps them photosynthesize.

Context2:
Photosynthesis in plants typically occurs in the leaves where chloroplasts are concentrated.

Complicated Multi-Hop Question:
In which plant structures does the pigment responsible for their verdancy facilitate energy production?

Initial Question:
How do you calculate the area of a rectangle?

Context1:
The area of a shape is calculated based on the shape's dimensions. For rectangles, this involves multiplying the length and width.

Context2:
Rectangles have four sides with opposite sides being equal in length. They are a type of quadrilateral.

Complicated Multi-Hop Question:
What multiplication involving equal opposites yields a quadrilateral's area?


Initial Question:
{question}
context1:
{context1}
context2:
{context2}
Complicated Multi-Hop Question:
"""  # noqa: E501
)


CONDITIONAL_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Rewrite the provided question to increase its complexity by introducing a conditional element.
The goal is to make the question more intricate by incorporating a scenario or condition that affects the context of the question.
Follow the rules given below while rewriting the question.
    1. The rewritten question should not be longer than 25 words. Use abbreviation wherever possible.
    2. The rewritten question must be reasonable and must be understood and responded by humans.
    3. The rewritten question must be fully answerable from information present context.
    4. phrases like 'provided context','according to the context?',etc are not allowed to appear in the question.

Initial Question:
What is the function of the roots of a plant?

Context:
The roots of a plant absorb water and nutrients from the soil, anchor the plant in the ground, and store food.

Rewritten Question:
What dual purpose do plant roots serve concerning soil nutrients and stability?

Answer:
Plant roots serve a dual purpose by absorbing water and nutrients from the soil, which is vital for the plant's growth, and providing stability by anchoring the plant in the ground.

Example 2:

Initial Question:
How do vaccines protect against diseases?

Context:
Vaccines protect against diseases by stimulating the body's immune response to produce antibodies, which recognize and combat pathogens.

Rewritten Question:
How do vaccines utilize the body's immune system to defend against pathogens?

Initial Question::
{question}
Context:
{context}
Rewritten Question
"""  # noqa: E501
)


COMPRESS_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Rewrite the following question to make it more indirect and shorter while retaining the essence of the original question. 
The goal is to create a question that conveys the same meaning but in a less direct manner. The rewritten question should shorter so use abbreviation wherever possible.

Original Question:
What is the distance between the Earth and the Moon?

Rewritten Question:
How far is the Moon from Earth?

Original Question:
What ingredients are required to bake a chocolate cake?

Rewritten Question:
What's needed for a chocolate cake?

Original Question:
{question}
Rewritten Question:
"""  # noqa: E501
)


CONVERSATION_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Reformat the provided question into two separate questions as if it were to be part of a conversation. Each question should focus on a specific aspect or subtopic related to the original question.
question: What are the advantages and disadvantages of remote work?
Reformatted Questions for Conversation: What are the benefits of remote work?\nOn the flip side, what challenges are encountered when working remotely?
question:{question}

Reformatted Questions for Conversation:
"""  # noqa: E501
)

SCORE_CONTEXT = HumanMessagePromptTemplate.from_template(
    """Given a context, complete the two following tasks and output answer valid json format 
1.Evaluate the provided context and assign a numerical score between 0 and 10 based on the following criteria:
    - Award a high score to context that thoroughly delves into and explains concepts.
    - Assign a lower score to context that contains excessive references, acknowledgments, personal information, or other non-essential elements.
2.Check if context contains tables
Context:
Albert Einstein (/ˈaɪnstaɪn/ EYEN-styne;[4] German: [ˈalbɛɐt ˈʔaɪnʃtaɪn] ⓘ; 14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.
Output:
{{"score":6.0, "is_table_present":false}}
Context:
{context}
Output:"""  # noqa: E501
)

REWRITE_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Given a context, transform the given question to be clear and standalone by replacing its coreferences with specific details from the context:

Contexts:
The Eiffel Tower was constructed using iron and was originally intended as a temporary exhibit for the 1889 World's Fair held in Paris.
Despite its initial temporary purpose, the Eiffel Tower quickly became a symbol of Parisian ingenuity and an iconic landmark of the city, attracting millions of visitors each year
The tower's design, created by Gustave Eiffel, was initially met with criticism from some French artists and intellectuals, but it has since been celebrated as a masterpiece of structural engineering and architectural design.
Question:
Who created the design for the Tower?
Rewritten question:
Who created the design for the Eiffel Tower?

Contexts:
'Exploring Zero-Shot Learning in Neural Networks' was published by Smith and Lee in 2021, focusing on the application of zero-shot learning techniques in artificial intelligence. 
Question: 
What datasets were used for the zero-shot evaluations in this study?
Rewritten question:
What datasets were used for the zero-shot evaluations Exploring Zero-Shot Learning in Neural Networks paper?


Question:{question}
Context: {context}
Rewritten question:
"""
)

FILTER_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Classify given question

question: What is the discovery about space?
{{
    "reason":"The question is too vague and does not specify which discovery about space it is referring to."
    "verdit":"No"
}}

question: What caused the Great Depression?
{{
    "reason":"The question is specific and refers to a well-known historical economic event, making it clear and answerable.",
    "verdict":"Yes"
}}

question: What is the keyword that best describes the paper's focus in natural language understanding tasks?
{{
  "reason": "The question mentions a 'paper' in it without referring it's name which makes it unclear without it",
  "verdict": "No"
}}
question: Who wrote 'Romeo and Juliet'?
{{
  "reason": "The question is clear and refers to a specific work by name therefore it is clear",
  "verdict": "Yes"
}}
question: What did the study mention?
{{
  "reason": "The question is vague and does not specify which study it is referring to",
  "verdict": "No"
}}
question: What is the focus of the REPLUG paper?
{{
    "reason": "The question refers to a specific work by it's name hence can be understood", 
    "verdict": "Yes"
}}

question: What is the purpose of the reward-driven stage in the training process?
{{
"reason": "The question lacks specific context regarding the type of training process, making it potentially ambiguous and open to multiple interpretations.",
"verdict": "No"
}}


question: {question}"""  # noqa: E501
)

EVOLUTION_ELIMINATION = HumanMessagePromptTemplate.from_template(
    """\
Check if the given two questions are equal based on following requirements:
1. They have same constraints and requirements.
2. They have same depth and breadth of the inquiry.

Question 1: What are the primary causes of climate change?
Question 2: What factors contribute to global warming?
{{
  "reason": "While both questions deal with environmental issues, 'climate change' encompasses broader changes than 'global warming', leading to different depths of inquiry.",
  "verdict": "Not Equal"
}}
Question 1: How does photosynthesis work in plants?
Question 2: Can you explain the process of photosynthesis in plants?
{{
  "reason": "Both questions ask for an explanation of the photosynthesis process in plants, sharing the same depth, breadth, and requirements for the answer.",
  "verdict": "Equal"
}}
Question 1: {question1}
Question 2: {question2}"""  # noqa: E501
)

ANSWER_FORMULATE = HumanMessagePromptTemplate.from_template(
    """\
Answer the question using the information from the given context. 
question:{question}
context:{context}
answer:
"""  # noqa: E501
)


INFORMAL_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Rewrite the following question into a casual, conversational form as if it's being asked by someone in an informal setting. 
Keep the core information request intact, without including any additional details or questions.
Formal Question: What are the criteria for Objectives and Key Results?
Casual Rewrite: What should I be looking at when I'm setting up OKRs?
Formal Question: Could you delineate the primary responsibilities of a project manager?
Casual Rewrite: What's the main job of a project manager, in simple terms?
Formal Question: What mechanisms underlie the process of cellular respiration?
Casual Rewrite: How does cellular respiration actually work?
Formal Question:{question}
Casual Rewrite:"""
)

CONTEXT_FORMULATE = HumanMessagePromptTemplate.from_template(
    """Please extract relevant sentences from the provided context that can potentially help answer the following question. While extracting candidate sentences you're not allowed to make any changes to sentences from given context.

question:{question}
context:\n{context}
candidate sentences:\n
"""  # noqa: E501
)


demonstrations = [
    {
        "context": "The Eiffel Tower in Paris was originally intended as a temporary structure, built for the 1889 World's Fair. It was almost dismantled in 1909 but was saved because it was repurposed as a giant radio antenna.",
        "questions": [
            {
                "question_Why": "Why was the Eiffel Tower originally planned to be a temporary structure?"
            },
            {
                "question_Was": "Was the Eiffel Tower originally designed to be a permanent structure?"
            },
            {
                "question_What": "What was the original purpose of the Eiffel Tower when it was built for the 1889 World's Fair?"
            },
            {
                "question_How": "How did the Eiffel Tower avoid being dismantled in 1909?"
            },
            {"question_Where": "Where is the Eiffel Tower?"},
        ],
    },
    {
        "context": "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
        "questions": [
            {"question_Why": "Why do plants perform photosynthesis?"},
            {
                "question_Was": "Was photosynthesis discovered in plants, algae, or bacteria first?"
            },
            {
                "question_What": "What converts light energy into chemical energy in photosynthesis?"
            },
            {"question_How": "How do plants capture light energy for photosynthesis?"},
            {"question_Where": "Where in plants does photosynthesis primarily occur?"},
            {"question_Can": "Can photosynthesis occur in the absence of light?"},
        ],
    },
]
