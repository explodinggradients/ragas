from ragas.llms.prompt import Prompt

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
    instruction="""Given a question, classify it based on clarity and specificity. Use only 'Clear' (1) and 'Unclear' (0) as verdict.""",
    examples=[
        {
            "question": "What is the discovery about space?",
            "output": {
                "reason": "The question is too vague and does not specify which discovery about space it is referring to.",
                "verdit": "0",
            },
        },
        {
            "question": "What caused the Great Depression?",
            "output": {
                "reason": "The question is specific and refers to a well-known historical economic event, making it clear and answerable.",
                "verdict": "1",
            },
        },
        {
            "question": "What is the keyword that best describes the paper's focus in natural language understanding tasks?",
            "output": {
                "reason": "The question mentions a 'paper' in it without referring it's name which makes it unclear without it",
                "verdict": "0",
            },
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "output": {
                "reason": "The question is clear and refers to a specific work by name therefore it is clear",
                "verdict": "1",
            },
        },
        {
            "question": "What did the study mention?",
            "output": {
                "reason": "The question is vague and does not specify which study it is referring to",
                "verdict": "0",
            },
        },
        {
            "question": "What is the focus of the REPLUG paper?",
            "output": {
                "reason": "The question refers to a specific work by it's name hence can be understood",
                "verdict": "1",
            },
        },
        {
            "question": "What is the purpose of the reward-driven stage in the training process?",
            "output": {
                "reason": "The question lacks specific context regarding the type of training process, making it potentially ambiguous and open to multiple interpretations.",
                "verdict": "0",
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
    2. They have same depth and breadth of the inquiry.
    Output verdict as 1 if they are equal and 0 if they are not""",
    examples=[
        {
            "question1": "What are the primary causes of climate change?",
            "question2": "What factors contribute to global warming?",
            "output": {
                "reason": "While both questions deal with environmental issues, 'climate change' encompasses broader changes than 'global warming', leading to different depths of inquiry.",
                "verdict": "0",
            },
        },
        {
            "question1": "How does photosynthesis work in plants?",
            "question2": "Can you explain the process of photosynthesis in plants?",
            "output": {
                "reason": "Both questions ask for an explanation of the photosynthesis process in plants, sharing the same depth, breadth, and requirements for the answer.",
                "verdict": "1",
            },
        },
        {
            "question1": "What are the health benefits of regular exercise?",
            "question2": "Can you list the advantages of exercising regularly for health?",
            "output": {
                "reason": "Both questions seek information about the positive effects of regular exercise on health. They require a similar level of detail in listing the health benefits.",
                "verdict": "1",
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

keyphrase_extraction_prompt = Prompt(
    name="keyphrase_extraction",
    instruction="Extract the top 3 to 5 keyphrases from the provided text, focusing on the most significant and distinctive aspects. ",
    examples=[
        {
            "text": "A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, has enough energy to escape it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.",
            "output": {
                "keyphrases": [
                    "Black hole",
                    "Region of spacetime",
                    "Strong gravity",
                    "Light and electromagnetic waves",
                    "Theory of general relativity",
                ]
            },
        },
        {
            "text": "The Great Wall of China is an ancient series of walls and fortifications located in northern China, built around 500 years ago. This immense wall stretches over 13,000 miles and is a testament to the skill and persistence of ancient Chinese engineers.",
            "output": {
                "keyphrases": [
                    "Great Wall of China",
                    "Ancient fortifications",
                    "Northern China",
                ]
            },
        },
    ],
    input_keys=["text"],
    output_key="output",
    output_type="json",
)


seed_question_prompt = Prompt(
    name="seed_question",
    instruction="generate a question that can be fully answered from given context. The question should contain atleast two of the given keyphrases",
    examples=[
        {
            "context": "Photosynthesis in plants involves converting light energy into chemical energy, using chlorophyll and other pigments to absorb light. This process is crucial for plant growth and the production of oxygen.",
            "keyphrases": [
                "Photosynthesis",
                "Light energy",
                "Chlorophyll",
                "Oxygen production",
            ],
            "question": "How does chlorophyll aid in converting light energy into chemical energy during photosynthesis?",
        },
        {
            "context": "The Industrial Revolution, starting in the 18th century, marked a major turning point in history as it led to the development of factories and urbanization.",
            "keyphrases": [
                "Industrial Revolution",
                "18th century",
                "Factories",
                "Urbanization",
            ],
            "question": "Why did the Industrial Revolution significantly contribute to the development of factories and urbanization?",
        },
        {
            "context": "A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, has enough energy to escape it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.",
            "keyphrases": [
                "Black hole",
                "region of spacetime",
                "Sufficiently compact mass",
                "Energy to escape",
            ],
            "question": "What is a black hole and how does it relate to a region of spacetime?",
        },
    ],
    input_keys=["context", "keyphrases"],
    output_key="question",
    output_type="string",
)

find_relevent_context_prompt = Prompt(
    name="find_relevent_context",
    instruction="Given a question and set of contexts, find the most relevant contexts to answer the question.",
    examples=[
        {
            "question": "What is the capital of France?",
            "contexts": [
                "1. France is a country in Western Europe. It has several cities, including Paris, Lyon, and Marseille. Paris is not only known for its cultural landmarks like the Eiffel Tower and the Louvre Museum but also as the administrative center.",
                "2. The capital of France is Paris. It is also the most populous city in France, with a population of over 2 million people. Paris is known for its cultural landmarks like the Eiffel Tower and the Louvre Museum.",
                "3. Paris is the capital of France. It is also the most populous city in France, with a population of over 2 million people. Paris is known for its cultural landmarks like the Eiffel Tower and the Louvre Museum.",
            ],
            "output": {
                "relevent_contexts": [1, 2],
            },
        },
        {
            "question": "How does caffeine affect the body and what are its common sources?",
            "contexts": [
                "1. Caffeine is a central nervous system stimulant. It can temporarily ward off drowsiness and restore alertness. It primarily affects the brain, where it alters the function of neurotransmitters.",
                "2. Regular physical activity is essential for maintaining good health. It can help control weight, combat health conditions, boost energy, and promote better sleep.",
                "3. Common sources of caffeine include coffee, tea, cola, and energy drinks. These beverages are consumed worldwide and are known for providing a quick boost of energy.",
            ],
            "output": {"relevant_contexts": [1, 2]},
        },
    ],
    input_keys=["question", "contexts"],
    output_key="output",
    output_type="json",
    language="english",
)
