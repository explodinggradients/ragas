from langchain.prompts import HumanMessagePromptTemplate

SEED_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Your task is to generate a question from given context satisfying the rules given below:
    1.The question should make sense to humans even when read without the given context.
    2.The question should be fully answered from the given context.
    3.The question should be framed from a part of context that contains important information. It can also be from tables,code,etc.
    4.The question should be framed as person would be asking using first person pronouns. 
    5.The question should be of moderate difficulty.
    6.The question must be reasonable and must be understood and responded by humans.
    7.Do no use phrases like 'provided context',etc in the question
    8.Avoid framing question using word "and" that can be decomposed into more than one question.
    9.The question should not contain more than 10 words, make of use of abbreviation wherever possible.

Context:
Mars is known as the Red Planet due to its reddish appearance, which is the result of iron oxide, commonly known as rust, on its surface.  
Question: 
Why is Mars called the Red Planet?

Context:
The Battle of Hastings in 1066 was significant because it led to the Norman conquest of England. This event dramatically altered English culture, language, and governance  
Question:
What is the significance of the Battle of Hastings?

Context:
The Eiffel Tower was constructed using iron and was originally intended as a temporary exhibit for the 1889 World's Fair held in Paris.
Question:
How was the Eiffel Tower originally intended?

Context:
The human heart, located between the lungs, functions as the circulatory system's primary pump, circulating blood throughout the body.
Question:
What does quantum mechanics describe?


context:{context}
"""  # noqa: E501
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

FILTER_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Determine if the given question can be clearly understood even when presented without any additional context. Specify reason and verdict is a valid json format.
question: 
What is the keyword that best describes the paper's focus in natural language understanding tasks?
{{"reason":"The specific paper being referred to is not mentioned in the question.", "verdict": "No"}}
question:
Who wrote 'Romeo and Juliet'?
{{"reason": "The question is clear and can be answered without additional context because the author of 'Romeo and Juliet' is a well-established historical}}
question:
{question}
"""  # noqa: E501
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
