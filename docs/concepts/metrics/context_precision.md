# Context Precision


This metric gauges the precision of the retrieved context, calculated based on both the `question` and `contexts`. The values fall within the range of (0, 1), with higher values indicating better precision.

Ideally, the retrieved context should exclusively contain essential information to address the provided query. To compute this, we initially estimate the value of $|S|$ by identifying sentences within the retrieved context that are relevant for answering the given question. The final score is determined by the following formula:

```{math}
:label: context_precision
\text{context precision} = {|S| \over |\text{Total number of sentences in retrived context}|}
```

```{hint}
Question: What is the capital of France?

High context precision: France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower. 

Low context precision: France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower. The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and the vast Palace of Versailles attest to its rich history.

```


## Example

```{code-block} python
:caption: Context precision using cross-encoder/nli-deberta-v3-xsmall 
from ragas.metrics import ContextPrecision
context_precision = ContextPrecision(
    model_name="cross-encoder/nli-deberta-v3-xsmall"
    )

# run init models to load the models used
context_precision.init_model()

# Dataset({
#     features: ['question','contexts'],
#     num_rows: 25
# })
dataset: Dataset

results = context_precision.score(dataset)
```