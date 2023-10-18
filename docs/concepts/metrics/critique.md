# Aspect Critique


This is designed to assess submissions based on predefined aspects such as `harmlessness` and `correctness`. Additionally, users have the flexibility to define their own aspects for evaluating submissions according to their specific criteria. The output of aspect critiques is binary, indicating whether the submission aligns with the defined aspect or not. This evaluation is performed using the 'answer' as input.

Critiques within the LLM evaluators evaluate submissions based on the provided aspect. Ragas Critiques offers a range of predefined aspects like correctness, harmfulness, etc. (Please refer to `SUPPORTED_ASPECTS` for a complete list). If you prefer, you can also create custom aspects to evaluate submissions according to your unique requirements.

The `strictness` parameter plays a crucial role in maintaining a certain level of self-consistency in predictions, with an ideal range typically falling between 2 to 4. It's important to note that the scores obtained from aspect critiques are binary and do not contribute to the final Ragas score due to their non-continuous nature.


```{hint}
SUPPORTED_ASPECTS = [
    harmfulness,
    maliciousness,
    coherence,
    correctness,
    conciseness,
]
```

## Example

```{code-block} python
:caption: Answer critique with custom definition
from ragas.metrics.critique import conciseness

# Dataset({
#     features: ['question','answer'],
#     num_rows: 25
# })
dataset: Dataset


# Define your critique
from ragas.metrics.critique import AspectCritique
my_critique = AspectCritique(name="my-critique", definition="Is the submission safe to children?", strictness=2)

results = my_critique.score(dataset)
```