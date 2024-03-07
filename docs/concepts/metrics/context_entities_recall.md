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
from datasets import Dataset 
from ragas.metrics import context_entity_recall
from ragas import evaluate

data_samples = {
    'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[context_entity_recall])
score.to_pandas()
```

## Calculation

Let us consider the ground truth and the contexts given above.

- **Step-1**: Find entities present in the ground truths.
    - Entities in ground truth (GE) - ['Taj Mahal', 'Yamuna', 'Agra', '1631', 'Shah Jahan', 'Mumtaz Mahal']
- **Step-2**: Find entities present in the context.
    - Entities in context (CE1) - ['Taj Mahal', 'Agra', 'Shah Jahan', 'Mumtaz Mahal', 'India']
    - Entities in context (CE2) - ['Taj Mahal', 'UNESCO', 'India']
- **Step-3**: Use the formula given above to calculate entity-recall
    ```{math}
    :label: context_entity_recall
    \text{context entity recall - 1} = \frac{| CE1 \cap GE |}{| GE |}
                                 = 4/6
                                 = 0.666
    ```

    ```{math}
    :label: context_entity_recall
    \text{context entity recall - 2} = \frac{| CE2 \cap GE |}{| GE |}
                                 = 1/6
                                 = 0.166
    ```

    We can see that the first context had a high entity recall, because it has a better entity coverage given the ground truth. If these two contexts were fetched by two retrieval mechanisms on same set of documents, we could say that the first mechanism was better than the other in use-cases where entities are of importance.

