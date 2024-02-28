# Context entities recall

This metric gives the measure of recall of the retrieved context, based on the number of entities present in both `ground_truths` and `contexts` relative to the number of entities present in the `ground_truths` alone. Simply put, it is a measure of what fraction of entities are recalled from `ground_truths`. This metric is useful in fact-based use cases like tourism help desk, historical QA, etc. This metric can help evaluate the retrieval mechanism for entities, based on comparison with entities present in `ground_truths`, because in cases where entities matter, we need the `contexts` which cover them.

To compute this metric, we use two sets, $GE$ and $CE$, as set of entities present in `ground_truths` and set of entities present in `contexts` respectively. We then take the number of elements in intersection of these sets and divide it by the number of elements present in the $GE$, given by the formula:

```{math}
:label: context_entity_recall
\text{context entity recall} = \frac{| CE \cap GE |}{| GE |}
````

```{hint}
**Ground truth**: The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in the Indian city of Agra. It was commissioned in 1631 by the Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal.

**High entity recall context**: The Taj Mahal is a symbol of love and architectural marvel located in Agra, India. It was built by the Mughal emperor Shah Jahan in memory of his beloved wife, Mumtaz Mahal. The structure is renowned for its intricate marble work and beautiful gardens surrounding it.

**Low entity recall context**: The Taj Mahal is an iconic monument in India. It is a UNESCO World Heritage Site and attracts millions of visitors annually. The intricate carvings and stunning architecture make it a must-visit destination.

````


## Example

```{code-block} python
from ragas.metrics import ContextEntityRecall
context_entity_recall = ContextEntityRecall()

# Dataset({
#     features: ['ground_truths','contexts'],
#     num_rows: 25
# })
dataset: Dataset

results = context_entity_recall.score(dataset)
```