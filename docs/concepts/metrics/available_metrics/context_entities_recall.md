## Context Entities Recall

`ContextEntityRecall` metric gives the measure of recall of the retrieved context, based on the number of entities present in both `reference` and `retrieved_contexts` relative to the number of entities present in the `reference` alone. Simply put, it is a measure of what fraction of entities are recalled from `reference`. This metric is useful in fact-based use cases like tourism help desk, historical QA, etc. This metric can help evaluate the retrieval mechanism for entities, based on comparison with entities present in `reference`, because in cases where entities matter, we need the `retrieved_contexts` which cover them.

To compute this metric, we use two sets, $GE$ and $CE$, as set of entities present in `reference` and set of entities present in `retrieved_contexts` respectively. We then take the number of elements in intersection of these sets and divide it by the number of elements present in the $GE$, given by the formula:

$$
\text{context entity recall} = \frac{| CE \cap GE |}{| GE |}
$$


### Example

```python
from ragas import SingleTurnSample
from ragas.metrics import ContextEntityRecall

sample = SingleTurnSample(
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."], 
)

scorer = ContextEntityRecall()

await scorer.single_turn_ascore(sample)
```

### How It’s Calculated



!!! example
    **reference**: The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in the Indian city of Agra. It was commissioned in 1631 by the Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal.
    **High entity recall context**: The Taj Mahal is a symbol of love and architectural marvel located in Agra, India. It was built by the Mughal emperor Shah Jahan in memory of his beloved wife, Mumtaz Mahal. The structure is renowned for its intricate marble work and beautiful gardens surrounding it.
    **Low entity recall context**: The Taj Mahal is an iconic monument in India. It is a UNESCO World Heritage Site and attracts millions of visitors annually. The intricate carvings and stunning architecture make it a must-visit destination.

Let us consider the ground truth and the contexts given above.

- **Step-1**: Find entities present in the ground truths.
    - Entities in ground truth (GE) - ['Taj Mahal', 'Yamuna', 'Agra', '1631', 'Shah Jahan', 'Mumtaz Mahal']
- **Step-2**: Find entities present in the context.
    - Entities in context (CE1) - ['Taj Mahal', 'Agra', 'Shah Jahan', 'Mumtaz Mahal', 'India']
    - Entities in context (CE2) - ['Taj Mahal', 'UNESCO', 'India']
- **Step-3**: Use the formula given above to calculate entity-recall
    
    $$
    \text{context entity recall 1} = \frac{| CE1 \cap GE |}{| GE |}
                                 = 4/6
                                 = 0.666
    $$

    $$
    \text{context entity recall 2} = \frac{| CE2 \cap GE |}{| GE |}
                                 = 1/6
    $$

    We can see that the first context had a high entity recall, because it has a better entity coverage given the ground truth. If these two contexts were fetched by two retrieval mechanisms on same set of documents, we could say that the first mechanism was better than the other in use-cases where entities are of importance.

