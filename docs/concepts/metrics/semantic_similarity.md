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
from ragas.metrics import AnswerSimilarity()
answer_similarity = AnswerSimilarity()


# Dataset({
#     features: ['answer','ground_truths'],
#     num_rows: 25
# })
dataset: Dataset

results = answer_similarity.score(dataset)
```