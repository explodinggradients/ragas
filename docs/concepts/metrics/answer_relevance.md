# Answer Relevance

The evaluation metric, Answer Relevancy, focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information. This metric is computed using the `question` and the `answer`, with values ranging between 0 and 1, where higher scores indicate better relevancy.

An answer is deemed relevant when it directly and appropriately addresses the original question. Importantly, our assessment of answer relevance does not consider factuality but instead penalizes cases where the answer lacks completeness or contains redundant details. To calculate this score, the LLM is prompted to generate an appropriate question for the generated answer multiple times, and the mean cosine similarity between these generated questions and the original question is measured. The underlying idea is that if the generated answer accurately addresses the initial question, the LLM should be able to generate questions from the answer that align with the original question.

```{hint}

Question: Where is France and what is it's capital?

Low relevance answer: France is in western Europe.

High relevance answer: France is in western Europe and Paris is its capital.
```

## Example

```{code-block} python
:caption: Answer relevancy with bge-base embeddings
from ragas.metrics import AnswerRelevancy
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings('BAAI/bge-base-en')
answer_relevancy = AnswerRelevancy(
    embeddings=embeddings
)

# init_model to load models used
answer_relevancy.init_model()

# Dataset({
#     features: ['question','answer'],
#     num_rows: 25
# })
dataset: Dataset

results = answer_relevancy.score(dataset)

```