# Faithfulness

This measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better.

The generated answer is regarded as faithful if all the claims that are made in the answer can be inferred from the given context. To calculate this a set of claims from the generated answer is first identified. Then each one of these claims are cross checked with given context to determine if it can be inferred from given context or not. The faithfulness score is given by divided by

```{math}
:label: faithfulness
\text{Faithfulness score} = {|\text{Number of claims that can be inferred from given context}| \over |\text{Total number of claims in the generated answer}|}
```


```{hint}
**Question**: Where and where was Einstein born?

**Context**: Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time

**High faithfulness answer**: Einstein was born in Germany on 14th March 1879.

**Low faithfulness answer**:  Einstein was born in Germany on 20th March 1879.
```


## Example

```{code-block} python
:caption: Faithfulness metric with batch size 10
from ragas.metrics.faithfulness import Faithfulness
faithfulness = Faithfulness(
    batch_size = 10
)
# Dataset({
#     features: ['question','contexts','answer'],
#     num_rows: 25
# })
dataset: Dataset

results = faithfulness.score(dataset)
```


