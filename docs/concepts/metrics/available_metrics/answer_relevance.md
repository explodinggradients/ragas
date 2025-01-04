## Response Relevancy

`ResponseRelevancy` metric focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information and higher scores indicate better relevancy. This metric is computed using the `user_input`, the `retrived_contexts` and the `response`. 

The Answer Relevancy is defined as the mean cosine similarity of the original `user_input` to a number of artificial questions, which where generated (reverse engineered) based on the `response`: 

$$
\text{answer relevancy} = \frac{1}{N} \sum_{i=1}^{N} cos(E_{g_i}, E_o)
$$

$$
\text{answer relevancy} = \frac{1}{N} \sum_{i=1}^{N} \frac{E_{g_i} \cdot E_o}{\|E_{g_i}\|\|E_o\|}
$$

Where: 

* $E_{g_i}$ is the embedding of the generated question $i$.
* $E_o$ is the embedding of the original question.
* $N$ is the number of generated questions, which is 3 default.

Please note, that even though in practice the score will range between 0 and 1 most of the time, this is not mathematically guaranteed, due to the nature of the cosine similarity ranging from -1 to 1.

An answer is deemed relevant when it directly and appropriately addresses the original question. Importantly, our assessment of answer relevance does not consider factuality but instead penalizes cases where the answer lacks completeness or contains redundant details. To calculate this score, the LLM is prompted to generate an appropriate question for the generated answer multiple times, and the mean cosine similarity between these generated questions and the original question is measured. The underlying idea is that if the generated answer accurately addresses the initial question, the LLM should be able to generate questions from the answer that align with the original question.


### Example

```python
from ragas import SingleTurnSample 
from ragas.metrics import ResponseRelevancy

sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )

scorer = ResponseRelevancy()
await scorer.single_turn_ascore(sample)
```

### How It’s Calculated

!!! example
    Question: Where is France and what is it's capital?

    Low relevance answer: France is in western Europe.

    High relevance answer: France is in western Europe and Paris is its capital.

To calculate the relevance of the answer to the given question, we follow two steps:

- **Step 1:** Reverse-engineer 'n' variants of the question from the generated answer using a Large Language Model (LLM). 
  For instance, for the first answer, the LLM might generate the following possible questions:
    - *Question 1:* "In which part of Europe is France located?"
    - *Question 2:* "What is the geographical location of France within Europe?"
    - *Question 3:* "Can you identify the region of Europe where France is situated?"

- **Step 2:** Calculate the mean cosine similarity between the generated questions and the actual question.

The underlying concept is that if the answer correctly addresses the question, it is highly probable that the original question can be reconstructed solely from the answer.
