from ragas.llms.prompt import Prompt

summary_extactor_prompt = Prompt(
    name="summary_extractor",
    instruction="Summarize the given text in less than 10 sentences.",
    examples=[
        {
            "text": "Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations.",
            "summary": "AI is revolutionizing industries by automating tasks, analyzing data, and driving innovations like self-driving cars and personalized recommendations.",
        }
    ],
    input_keys=["text"],
    output_key="summary",
    output_type="str",
)

headline_extractor_prompt = Prompt(
    name="headline_extractor",
    instruction="Extract section titles and subtitles from the given text. The extracted headlines should be unique and match exactly as they appear in the text.",
    examples=[
        {
            "text": """
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
            "headlines": {
                "1. Introduction and Related Work": [
                    "1.1 Conditional Computation",
                    "1.2 Our Approach: The Sparsely-Gated Mixture-of-Experts Layer",
                    "1.3 Related Work on Mixtures of Experts",
                ],
                "2. The Sparsely-Gated Mixture-of-Experts Layer": ["2.1 Architecture"],
            },
        }
    ],
    input_keys=["text"],
    output_key="headlines",
    output_type="json",
    language="english",
)


keyphrase_extractor_prompt = Prompt(
    name="keyphrase_extractor",
    instruction="Extract top 5 keyphrases from the given text.",
    examples=[
        {
            "text": "Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations.",
            "keyphrases": [
                "Artificial intelligence",
                "automating tasks",
                "healthcare",
                "self-driving cars",
                "personalized recommendations",
            ],
        }
    ],
    input_keys=["text"],
    output_key="keyphrases",
    output_type="json",
)

title_extractor_prompt = Prompt(
    name="title_extractor",
    instruction="Extract the title of the given document.",
    examples=[
        {
            "text": "Deep Learning for Natural Language Processing\n\nAbstract\n\nDeep learning has revolutionized the field of natural language processing (NLP). This paper explores various deep learning models and their applications in NLP tasks such as language translation, sentiment analysis, and text generation. We discuss the advantages and limitations of different models, and provide insights into future research directions.",
            "title": "Deep Learning for Natural Language Processing",
        },
    ],
    input_keys=["text"],
    output_key="title",
    output_type="str",
    language="english",
)
