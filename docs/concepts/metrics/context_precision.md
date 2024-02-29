# Context Precision

Context Precision is a metric that evaluates whether all of the ground-truth relevant items present in the `contexts` are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. This metric is computed using the `question` and the `contexts`, with values ranging between 0 and 1, where higher scores indicate better precision.

```{math}
\text{Context Precision@k} = {\sum {\text{precision@k}} \over \text{total number of relevant items in the top K results}}
````

```{math}
\text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})}
````


Where k is the total number of chunks in `contexts`

```{hint}
Question: Where is France and what is it's capital?
Ground truth: France is in Western Europe and its capital is Paris.

High context precision: ["France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower", "The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and the vast Palace of Versailles attest to its rich history."]  

Low context precision: ["The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and", "France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower",]
```

:::{dropdown} How was this calculated?
Let's examine how context precision was calculated using the low context precision example:

**Step 1**: For each chunk in retrieved context, check if it is relevant or not relevant to arrive at the ground truth for the given question.

**Step 2**: Calculate precision@k for each chunk in the context.

```{math}
\text{Precision@1} = {\text{0} \over \text{1}} = 0
````

```{math}
\text{Precision@2} = {\text{1} \over \text{2}} = 0.5
````

**Step 3**: Calculate the mean of precision@k to arrive at the final context precision score.

```{math}
 \text{Context Precision} = {\text{(0+0.5)} \over \text{2}} = 0.25
```

:::
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