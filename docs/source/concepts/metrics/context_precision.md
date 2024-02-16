# Context Precision

Context Precision is a metric that evaluates whether all of the ground-truth relevant items present in the `contexts` are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. This metric is computed using the `question` and the `contexts`, with values ranging between 0 and 1, where higher scores indicate better precision.

```{math}
\text{Context Precision@k} = {\sum {\text{precision@k}} \over \text{total number of relevant items in the top K results}}
````

```{math}
\text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})}
````


Where k is the total number of chunks in `contexts`
## Example

```{code-block} python
:caption: Context precision
from ragas.metrics import ContextPrecision
context_precision = ContextPrecision()


# Dataset({
#     features: ['question','contexts'],
#     num_rows: 25
# })
dataset: Dataset

results = context_precision.score(dataset)
```