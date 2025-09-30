"""
Prompts for Ragas metrics.

This module contains simple string prompts used by various metrics.
These prompts use Python's .format() method for variable substitution.
"""

# Answer Relevancy Prompts

ANSWER_RELEVANCY_DIRECT_SCORING = """Rate how relevant the answer is to the question on a scale of 0.0 to 1.0.

Question: {question}
Answer: {answer}

Consider:
- Does the answer directly address the question?
- Is the answer complete and informative?
- Is the answer evasive, vague, or noncommittal?

Provide your response in the following format:
Score: [0.0 to 1.0]
Reasoning: [Brief explanation of your scoring]
Is_Noncommittal: [true/false - whether the answer is evasive or vague]

Examples:

Question: What is Python?
Answer: Python is a high-level programming language known for its simplicity and readability.
Score: 0.95
Reasoning: Answer directly addresses the question with accurate and informative content.
Is_Noncommittal: false

Question: How do you make coffee?
Answer: I'm not sure about coffee preparation methods.
Score: 0.1
Reasoning: Answer is evasive and doesn't provide useful information about coffee preparation.
Is_Noncommittal: true

Question: What's the weather like?
Answer: Python is a programming language used for software development.
Score: 0.0
Reasoning: Answer is completely irrelevant to the weather question.
Is_Noncommittal: false"""

# Context Precision Prompts (for future use)
CONTEXT_PRECISION_SCORING = """Rate how precise and relevant the given context is for answering the question on a scale of 0.0 to 1.0.

Question: {question}
Context: {context}

Consider:
- Does the context contain information directly relevant to the question?
- Is the context focused and not containing irrelevant information?
- How much of the context is actually useful for answering the question?

Provide your response in the following format:
Score: [0.0 to 1.0]
Reasoning: [Brief explanation of your scoring]"""

# Faithfulness Prompts (for future use)
FAITHFULNESS_SCORING = """Rate how faithful the answer is to the given context on a scale of 0.0 to 1.0.

Context: {context}
Answer: {answer}

Consider:
- Is the answer supported by the information in the context?
- Does the answer contain any information not present in the context?
- Are there any contradictions between the answer and the context?

Provide your response in the following format:
Score: [0.0 to 1.0]
Reasoning: [Brief explanation of your scoring]
Contains_Hallucination: [true/false - whether answer contains unsupported information]"""
