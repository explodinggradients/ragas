# Answer Relevance

The evaluation metric, Answer Relevancy, focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information. This metric is computed using the `question` and the `answer`, with values ranging between 0 and 1, where higher scores indicate better relevancy.

:::{note}
This is reference free metric. If you're looking to compare ground truth answer with generated answer refer to [answer_correctness](./answer_correctness.md)
:::

An answer is deemed relevant when it directly and appropriately addresses the original question. Importantly, our assessment of answer relevance does not consider factuality but instead penalizes cases where the answer lacks completeness or contains redundant details. To calculate this score, the LLM is prompted to generate an appropriate question for the generated answer multiple times, and the mean cosine similarity between these generated questions and the original question is measured. The underlying idea is that if the generated answer accurately addresses the initial question, the LLM should be able to generate questions from the answer that align with the original question.

```{hint}

Question: Where is France and what is it's capital?

Low relevance answer: France is in western Europe.

High relevance answer: France is in western Europe and Paris is its capital.
```

:::{dropdown} How is this calculated?
To calculate the relevance of the answer to the given question, we follow two steps:

- **Step 1:** Reverse-engineer 'n' variants of the question from the generated answer using a Large Language Model (LLM). 
  For instance, for the first answer, the LLM might generate the following possible questions:
    - *Question 1:* "In which part of Europe is France located?"
    - *Question 2:* "What is the geographical location of France within Europe?"
    - *Question 3:* "Can you identify the region of Europe where France is situated?"

- **Step 2:** Calculate the mean cosine similarity between the generated questions and the actual question.

The underlying concept is that if the answer correctly addresses the question, it is highly probable that the original question can be reconstructed solely from the answer.

:::

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