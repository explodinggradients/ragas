#  Semantic similarity

The concept of Answer Semantic Similarity pertains to the assessment of the semantic resemblance between the generated answer and the ground truth. This evaluation is based on the `ground truth` and the `answer`, with values falling within the range of 0 to 1. A higher score signifies a better alignment between the generated answer and the ground truth.

Measuring the semantic similarity between answers can offer valuable insights into the quality of the generated response. This evaluation utilizes a cross-encoder model to calculate the semantic similarity score.


### Example

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SemanticSimilarity


sample = SingleTurnSample(
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris. I has a height of 1000ft."
)

scorer = SemanticSimilarity()
scorer.embeddings = embedding_model
await scorer.single_turn_ascore(sample)

```

### How It’s Calculated 

!!! example

    **reference**: Albert Einstein's theory of relativity revolutionized our understanding of the universe."

    **High similarity answer**: Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos.

    **Low similarity answer**: Isaac Newton's laws of motion greatly influenced classical physics.

Let's examine how answer similarity was calculated for the first answer:

- **Step 1:** Vectorize the ground truth answer using the specified embedding model.
- **Step 2:** Vectorize the generated answer using the same embedding model.
- **Step 3:** Compute the cosine similarity between the two vectors.

