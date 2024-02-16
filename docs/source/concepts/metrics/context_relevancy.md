# Context Relevancy


This metric gauges the relevancy of the retrieved context, calculated based on both the `question` and `contexts`. The values fall within the range of (0, 1), with higher values indicating better relevancy.

Ideally, the retrieved context should exclusively contain essential information to address the provided query. To compute this, we initially estimate the value of $|S|$ by identifying sentences within the retrieved context that are relevant for answering the given question. The final score is determined by the following formula:

```{math}
:label: context_relevancy
\text{context relevancy} = {|S| \over |\text{Total number of sentences in retrived context}|}
```

```{hint}
Question: What is the capital of France?

High context relevancy: France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower. 

Low context relevancy: France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower. The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and the vast Palace of Versailles attest to its rich history.

```


## Example

```{code-block} python
:caption: Context relevancy
from ragas.metrics import ContextRelevance
context_relevancy = ContextRelevance()


# Dataset({
#     features: ['question','contexts'],
#     num_rows: 25
# })
dataset: Dataset

results = context_relevancy.score(dataset)
```