from ragas.llms.prompt import Prompt

common_theme_from_summaries = Prompt(
    name="get_common_themes",
    instruction="Analyze the following summaries and identify given number of common themes. The themes should be concise, descriptive, and highlight a key aspect shared across the summaries.",
    examples=[
        {
            "summaries": """
                Summary 1: Advances in artificial intelligence have revolutionized many industries. From healthcare to finance, AI algorithms are making processes more efficient and accurate. Machine learning models are being used to predict diseases, optimize investment strategies, and even recommend personalized content to users. The integration of AI into daily operations is becoming increasingly indispensable for modern businesses.

                Summary 2: The healthcare industry is witnessing a significant transformation due to AI advancements. AI-powered diagnostic tools are improving the accuracy of medical diagnoses, reducing human error, and enabling early detection of diseases. Additionally, AI is streamlining administrative tasks, allowing healthcare professionals to focus more on patient care. Personalized treatment plans driven by AI analytics are enhancing patient outcomes.

                Summary 3: Financial technology, or fintech, has seen a surge in AI applications. Algorithms for fraud detection, risk management, and automated trading are some of the key innovations in this sector. AI-driven analytics are helping companies to understand market trends better and make informed decisions. The use of AI in fintech is not only enhancing security but also increasing efficiency and profitability.
            """,
            "num_themes": 2,
            "themes": [
                {
                    "theme": "AI enhances efficiency and accuracy in various industries",
                    "description": "AI algorithms are improving processes across healthcare, finance, and more by increasing efficiency and accuracy.",
                },
                {
                    "theme": "AI-powered tools improve decision-making and outcomes",
                    "description": "AI applications in diagnostic tools, personalized treatment plans, and fintech analytics are enhancing decision-making and outcomes.",
                },
            ],
        }
    ],
    input_keys=["summaries", "num_themes"],
    output_key="themes",
    output_type="json",
    language="english",
)


common_topic_from_keyphrases = Prompt(
    name="get_common_topic",
    instruction="Identify a list of common concepts from the given list of key phrases for comparing the given theme across reports.",
    examples=[
        {
            "keyphrases": [
                ["fast charging", "long battery life", "OLED display", "waterproof"],
                ["quick charge", "extended battery", "HD display", "dust resistant"],
                ["rapid charging", "durable battery", "AMOLED display", "splash proof"],
                [
                    "fast charge",
                    "prolonged battery",
                    "retina display",
                    "water resistant",
                ],
            ],
            "num_concepts": 4,
            "concepts": [
                {
                    "Charging": [
                        "fast charging",
                        "quick charge",
                        "rapid charging",
                        "fast charge",
                    ]
                },
                {
                    "Battery Life": [
                        "long battery life",
                        "extended battery",
                        "durable battery",
                        "prolonged battery",
                    ]
                },
                {
                    "Display": [
                        "OLED display",
                        "HD display",
                        "AMOLED display",
                        "retina display",
                    ]
                },
                {
                    "Water/Dust Resistance": [
                        "waterproof",
                        "dust resistant",
                        "splash proof",
                        "water resistant",
                    ]
                },
            ],
        }
    ],
    input_keys=["keyphrases", "num_concepts"],
    output_key="concepts",
    output_type="json",
    language="english",
)


# comparative_question = Prompt(
#     name="comparative_question",
#     instruction="Generate a comparative question based on the given themes that can be answered by comparing the information in the provided contexts.",
#     examples=[
#         {
#             "themes": [
#                 "AI enhances efficiency and accuracy in various industries",
#                 "AI-powered tools improve decision-making and outcomes"
#             ],
#             "contexts": [
#                 """
#                 Context 1: Advances in artificial intelligence have revolutionized many industries. From healthcare to finance, AI algorithms are making processes more efficient and accurate. Machine learning models are being used to predict diseases, optimize investment strategies, and even recommend personalized content to users. The integration of AI into daily operations is becoming increasingly indispensable for modern businesses.
#                 """,
#                 """
#                 Context 2: The healthcare industry is witnessing a significant transformation due to AI advancements. AI-powered diagnostic tools are improving the accuracy of medical diagnoses, reducing human error, and enabling early detection of diseases. Additionally, AI is streamlining administrative tasks, allowing healthcare professionals to focus more on patient care. Personalized treatment plans driven by AI analytics are enhancing patient outcomes.
#                 """
#             ],
#             "question": "How do AI-powered tools improve decision-making and outcomes compared to enhancing efficiency and accuracy in various industries?",
#         }
#     ],
#     input_keys=["themes", "contexts"],
#     output_key="question",
#     output_type="str",
#     language="english",
# )

abstract_comparative_question = Prompt(
    name="create_comparative_question",
    instruction="Generate an abstract comparative question based on the given concept, keyphrases belonging to that concept, and summaries of reports.",
    examples=[
        {
            "concept": "Battery Life",
            "keyphrases": [
                "long battery life",
                "extended battery",
                "durable battery",
                "prolonged battery",
            ],
            "summaries": [
                "Report 1: The device offers a long battery life, capable of lasting up to 24 hours on a single charge.",
                "Report 2: Featuring an extended battery, the product can function for 20 hours with heavy usage.",
                "Report 3: With a durable battery, this model ensures 22 hours of operation under normal conditions.",
                "Report 4: The battery life is prolonged, allowing the gadget to be used for up to 18 hours on one charge.",
            ],
            "question": "How do the battery life claims and performance metrics compare across different reports for devices featuring long battery life, extended battery, durable battery, and prolonged battery?",
        }
    ],
    input_keys=["concept", "keyphrases", "summaries"],
    output_key="question",
    output_type="str",
    language="english",
)


abstract_question_from_theme = Prompt(
    name="abstract_question_generation",
    instruction="Generate an abstract conceptual question using the given theme that can be answered from the information in the provided context.",
    examples=[
        {
            "theme": "AI enhances efficiency and accuracy in various industries.",
            "context": """
            AI is transforming various industries by improving efficiency and accuracy. For instance, in manufacturing, AI-powered robots automate repetitive tasks with high precision, reducing errors and increasing productivity. In healthcare, AI algorithms analyze medical images and patient data to provide accurate diagnoses and personalized treatment plans. Financial services leverage AI for fraud detection and risk management, ensuring quicker and more reliable decision-making. Overall, AI's ability to process vast amounts of data and learn from it enables industries to optimize operations, reduce costs, and deliver better outcomes.
            """,
            "question": "How does AI improve efficiency and accuracy across different industries?",
        }
    ],
    input_keys=["theme", "context"],
    output_key="question",
    output_type="str",
    language="english",
)


critic_question = Prompt(
    name="critic_question",
    instruction="Critique the synthetically generated question based on the following rubrics. Provide a score for each rubric: Independence and Clear Intent. Scores are given as low (0), medium (1), or high (2).",
    examples=[
        {
            "question": "How does AI improve efficiency and accuracy across different industries?",
            "feedback": {"Independence": 2, "Clear Intent": 2},
        },
        {
            "question": "Explain the benefits of AI.",
            "feedback": {"Independence": 1, "Clear Intent": 1},
        },
        {
            "question": "How does AI?",
            "feedback": {"Independence": 0, "Clear Intent": 0},
        },
    ],
    input_keys=["question"],
    output_key="feedback",
    output_type="json",
    language="english",
)


question_answering = Prompt(
    name="question_answering",
    instruction="Answer the following question based on the information provided in the given text.",
    examples=[
        {
            "question": "How does AI improve efficiency and accuracy across different industries?",
            "text": """
                Advances in artificial intelligence have revolutionized many industries. From healthcare to finance, AI algorithms are making processes more efficient and accurate. Machine learning models are being used to predict diseases, optimize investment strategies, and even recommend personalized content to users. The integration of AI into daily operations is becoming increasingly indispensable for modern businesses.
            """,
            "answer": "AI improves efficiency and accuracy across different industries by making processes more efficient and accurate. In healthcare, AI predicts diseases and personalizes treatment plans. In finance, AI optimizes investment strategies and enhances fraud detection.",
        },
    ],
    input_keys=["question", "text"],
    output_key="answer",
    output_type="str",
    language="english",
)


question_modification = Prompt(
    name="question_modification",
    instruction="Modify the given question in order to fit the given style and length",
    examples=[],
    input_keys=["question", "style", "length"],
    output_key="modified_question",
    output_type="str",
)


EXAMPLES_FOR_QUESTION_MODIFICATION = [
    # Short Length Examples
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Misspelled queries",
        "length": "short",
        "modified_question": "How do enrgy storag solutions compare on efficincy?",
    },
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Perfect grammar",
        "length": "short",
        "modified_question": "How do energy storage solutions compare?",
    },
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Poor grammar",
        "length": "short",
        "modified_question": "How do storag solutions compare on efficiency?",
    },
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Web search like queries",
        "length": "short",
        "modified_question": "compare energy storage solutions efficiency cost sustainability",
    },
    # Medium Length Examples
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Misspelled queries",
        "length": "medium",
        "modified_question": "How do enrgy storag solutions compare on efficincy n cost?",
    },
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Perfect grammar",
        "length": "medium",
        "modified_question": "How do energy storage solutions compare in efficiency and cost?",
    },
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Poor grammar",
        "length": "medium",
        "modified_question": "How energy storag solutions compare on efficiency and cost?",
    },
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Web search like queries",
        "length": "medium",
        "modified_question": "comparison of energy storage solutions efficiency cost sustainability",
    },
    # Long Length Examples
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Misspelled queries",
        "length": "long",
        "modified_question": "How do various enrgy storag solutions compare in terms of efficincy, cost, and sustanbility in rnewable energy systems?",
    },
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Perfect grammar",
        "length": "long",
        "modified_question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
    },
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Poor grammar",
        "length": "long",
        "modified_question": "How various energy storag solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
    },
    {
        "question": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        "style": "Web search like queries",
        "length": "long",
        "modified_question": "How do various energy storage solutions compare efficiency cost sustainability renewable energy systems?",
    },
]

order_sections_by_relevance = order_sections_by_relevance = Prompt(
    name="order_sections_by_relevance",
    instruction="Classify each section based on its ability to create good specific questions from it, using 'low', 'medium', or 'high' as the classification.",
    examples=[
        {
            "sections": [
                "Abstract",
                "Introduction",
                "Literature Review",
                "Methodology",
                "Results",
                "Discussion",
                "Conclusion",
                "Future Work",
                "References",
                "Appendix",
            ],
            "ordered_sections": {
                "high": ["Methodology", "Results", "Discussion"],
                "medium": ["Literature Review", "Future Work", "Conclusion"],
                "low": ["Introduction", "Abstract", "References", "Appendix"],
            },
        }
    ],
    input_keys=["sections"],
    output_key="ordered_sections",
    output_type="json",
    language="english",
)


specific_question_from_keyphrase = Prompt(
    name="specific_question_from_keyphrase",
    instruction="Given the title of a text and a text chunk, along with a keyphrase from the chunk, generate a specific question related to the keyphrase.\n\n"
    "1. Read the title and the text chunk.\n"
    "2. Identify the context of the keyphrase within the text chunk.\n"
    "3. Formulate a question that directly relates to the keyphrase and its context within the chunk.\n"
    "4. Ensure the question is clear, specific, and relevant to the keyphrase.",
    examples=[
        {
            "title": "The Impact of Artificial Intelligence on Modern Healthcare",
            "keyphrase": "personalized treatment plans",
            "text": "Artificial intelligence (AI) is revolutionizing healthcare by improving diagnostic accuracy and enabling personalized treatment plans. AI algorithms analyze vast amounts of medical data to identify patterns and predict patient outcomes, which enhances the decision-making process for healthcare professionals.",
            "question": "How does artificial intelligence contribute to the development of personalized treatment plans in healthcare?",
        }
    ],
    input_keys=["title", "keyphrase", "text"],
    output_key="question",
    output_type="str",
)
