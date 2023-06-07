# Metrics

1. `factuality` : measures the factual consistency of the generated answer against the given context. This is done using a multi step paradigm that includes creation of statements from the generated answer followed by verifying each of these statements against the context. The answer is scaled to (0,1) range. Higher the better.

2. `answer_relevancy`: measures how relevant is the generated answer to the prompt. This is quantified using conditional likelihood of an LLM generating the question given the answer. This is implemented using a custom model. Values range (0,1), higher the better.

3. `context_relevancy`: measures how relevant is the retrieved context to the prompt. This is quantified using a custom trained cross encoder model. Values range (0,1), higher the better.