## MultiModalRelevance

`MultiModalRelevance` metric measures the relevance of the generated answer against both visual and textual context. It is calculated from the user input, response, and retrieved contexts (both visual and textual). The answer is scaled to a (0,1) range, with higher scores indicating better relevance.

The generated answer is regarded as relevant if it aligns with the visual or textual context provided. To determine this, the response is directly evaluated against the provided contexts, and the relevance score is either 0 or 1.

### Example

```python
from ragas.database_schema import SingleTurnSample 
from ragas.metrics import MultiModalRelevance

sample = SingleTurnSample(
        user_input="What about the Tesla Model X?",
        response="Cats are cute.",
        retrieved_contexts=[
            "custom_eval/multimodal/images/tesla.jpg"
        ]
    )
scorer = MultiModalRelevance()
await scorer.single_turn_ascore(sample)
```

### How It’s Calculated 

!!! example
    **Question**: What about the Tesla Model X?

    **Context (visual)**: 
    - An image of the Tesla Model X (custom_eval/multimodal/images/tesla.jpg)

    **High relevance answer**: The Tesla Model X is an electric SUV manufactured by Tesla.

    **Low relevance answer**: Cats are cute.

Let's examine how relevance was calculated using the low relevance answer:

- **Step 1:** Evaluate the generated response against the given contexts.
    - Response: "Cats are cute."

- **Step 2:** Verify if the response aligns with the given context.
    - Response: No

- **Step 3:** Use the result to determine the relevance score.

    $$
    \text{Relevance} = 0
    $$

In this example, the response "Cats are cute" does not align with the image of the Tesla Model X, so the relevance score is 0.