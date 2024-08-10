

# Context Recall

Context recall measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed using `question`, `ground truth` and the retrieved `context`, and the values range between 0 and 1, with higher values indicating better performance.
To estimate context recall from the ground truth answer, each claim in the ground truth answer is analyzed to determine whether it can be attributed to the retrieved context or not. In an ideal scenario, all claims in the ground truth answer should be attributable to the retrieved context.
A reference free version of this is available as [context_utilization](context_utilization.md).

The formula for calculating context recall is as follows:

```{math}
\text{context recall} = {|\text{GT claims that can be attributed to context}| \over |\text{Number of claims in GT}|}
```

```{hint}

Question: Where is France and what is it's capital?

Ground truth: France is in Western Europe and its capital is Paris. 

High context recall: France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower.

Low context recall: France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and the vast Palace of Versailles attest to its rich history.
```

## Example

```{code-block} python
:caption: Context recall
from datasets import Dataset 
from ragas.metrics import context_recall
from ragas import evaluate

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[context_recall])
score.to_pandas()
```

## Calculation

Let's examine how context recall was calculated using the low context recall example:

- **Step 1:** Break the ground truth answer into individual statements.
    - Statements:
        - Statement 1: "France is in Western Europe."
        - Statement 2: "Its capital is Paris."
- **Step 2:** For each of the ground truth statements, verify if it can be attributed to the retrieved context.
    - Statement 1: Yes
    - Statement 2: No

- **Step 3:** Use the formula depicted above to calculate context recall.
    ```{math}
    \text{context recall} = { \text{1} \over \text{2} } = 0.5
    ``` 

