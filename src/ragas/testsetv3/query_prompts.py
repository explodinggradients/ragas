from ragas.llms.prompt import Prompt

common_theme_from_summaries = Prompt(
    name="get_common_themes",
    instruction="Analyze the following summaries and identify one common theme. The theme should be concise, descriptive, and highlight a key aspect shared across the summaries.",
    examples=[
        {
            "summaries": """
                Summary 1: Advances in artificial intelligence have revolutionized many industries. From healthcare to finance, AI algorithms are making processes more efficient and accurate. Machine learning models are being used to predict diseases, optimize investment strategies, and even recommend personalized content to users. The integration of AI into daily operations is becoming increasingly indispensable for modern businesses.

                Summary 2: The healthcare industry is witnessing a significant transformation due to AI advancements. AI-powered diagnostic tools are improving the accuracy of medical diagnoses, reducing human error, and enabling early detection of diseases. Additionally, AI is streamlining administrative tasks, allowing healthcare professionals to focus more on patient care. Personalized treatment plans driven by AI analytics are enhancing patient outcomes.

                Summary 3: Financial technology, or fintech, has seen a surge in AI applications. Algorithms for fraud detection, risk management, and automated trading are some of the key innovations in this sector. AI-driven analytics are helping companies to understand market trends better and make informed decisions. The use of AI in fintech is not only enhancing security but also increasing efficiency and profitability.
            """,
            "theme": "AI enhances efficiency and accuracy in various industries.",
        }
    ],
    input_keys=["summaries"],
    output_key="theme",
    output_type="str",
    language="english",
)

common_topic_from_keyphrases = Prompt(
    name="get_common_topic",
    instruction="Identify a list of common topics from the given list of key phrases for comparing the given theme across reports.",
    examples=[
        {
            "theme": "Renewable Energy Technologies",
            "keyphrases": [
                "solar panels",
                "photovoltaic cells",
                "energy conversion efficiency",
                "wind turbines",
                "offshore wind farms",
                "renewable energy storage",
                "battery technology",
                "energy grid integration",
                "sustainable energy sources",
                "hydropower systems",
            ],
            "topics": [
                "Energy Conversion Efficiency",
                "Renewable Energy Storage",
                "Energy Grid Integration",
                "Sustainable Energy Sources",
            ],
        }
    ],
    input_keys=["theme", "keyphrases"],
    output_key="topics",
    output_type="json",
    language="english",
)


comparative_question = Prompt(
    name="comparative_question",
    instruction="Formulate a abstract comparative query based on the given theme and topic. The query should be aimed at comparing different aspects or techniques related to the theme.",
    examples=[
        {
            "theme": "Renewable Energy Technologies",
            "topic": "Energy Storage Solutions",
            "query": "How do various energy storage solutions compare in terms of efficiency, cost, and sustainability in renewable energy systems?",
        },
    ],
    input_keys=["theme", "topic"],
    output_key="query",
    output_type="str",
    language="english",
)

abstract_question_from_theme = Prompt(
    name="abstract_question_generation",
    instruction="Generate an abstract conceptual question for the given theme using that can be answered from the information in the provided summaries.",
    examples=[
        {
            "theme": "AI enhances efficiency and accuracy in various industries.",
            "summaries": """
                Summary 1: Advances in artificial intelligence have revolutionized many industries. From healthcare to finance, AI algorithms are making processes more efficient and accurate. Machine learning models are being used to predict diseases, optimize investment strategies, and even recommend personalized content to users. The integration of AI into daily operations is becoming increasingly indispensable for modern businesses.

                Summary 2: The healthcare industry is witnessing a significant transformation due to AI advancements. AI-powered diagnostic tools are improving the accuracy of medical diagnoses, reducing human error, and enabling early detection of diseases. Additionally, AI is streamlining administrative tasks, allowing healthcare professionals to focus more on patient care. Personalized treatment plans driven by AI analytics are enhancing patient outcomes.

                Summary 3: Financial technology, or fintech, has seen a surge in AI applications. Algorithms for fraud detection, risk management, and automated trading are some of the key innovations in this sector. AI-driven analytics are helping companies to understand market trends better and make informed decisions. The use of AI in fintech is not only enhancing security but also increasing efficiency and profitability.
            """,
            "question": "How does AI improve efficiency and accuracy across different industries?",
        }
    ],
    input_keys=["theme", "summaries"],
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
    examples=[
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
    ],
    input_keys=["question", "style", "length"],
    output_key="modified_question",
    output_type="str",
)
