# Answer semantic similarity


The concept of Answer Semantic Similarity pertains to the assessment of the semantic resemblance between the generated answer and the ground truth. This evaluation is based on the `ground truth` and the `answer`, with values falling within the range of 0 to 1. A higher score signifies a better alignment between the generated answer and the ground truth.

Measuring the semantic similarity between answers can offer valuable insights into the quality of the generated response. This evaluation utilizes a cross-encoder model to calculate the semantic similarity score.


```{hint}
Ground truth: Albert Einstein's theory of relativity revolutionized our understanding of the universe."

High similarity answer: Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos.

Low similarity answer: Isaac Newton's laws of motion greatly influenced classical physics.

```

## Example

```{code-block} python
from datasets import Dataset 
from ragas.metrics import answer_similarity
from ragas import evaluate


data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[answer_similarity])
score.to_pandas()
```

## Calculation 

Let's examine how answer similarity was calculated for the first answer:

- **Step 1:** Vectorize the ground truth answer using the specified embedding model.
- **Step 2:** Vectorize the generated answer using the same embedding model.
- **Step 3:** Compute the cosine similarity between the two vectors.

