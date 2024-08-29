# Context utilization

Context utilization is like a reference free version of [context_precision](context_precision.md) metrics. Context utilization is a metric that evaluates whether all of the answer relevant items present in the `contexts` are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. This metric is computed using the `question`, `answer` and the `contexts`, with values ranging between 0 and 1, where higher scores indicate better precision. 


```{math}
\text{Context Utilization@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant items in the top } K \text{ results}}
````

```{math}
\text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})}
````


Where $K$ is the total number of chunks in `contexts` and $v_k \in \{0, 1\}$ is the relevance indicator at rank $k$.

```{hint}
Question: Where is France and what is it's capital?
Answer: France is in Western Europe and its capital is Paris.

High context utilization: ["France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower", "The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and the vast Palace of Versailles attest to its rich history."]  

Low context utilization: ["The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and", "France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower",]
```

## Example

```{code-block} python
:caption: Context precision
from datasets import Dataset 
from ragas.metrics import context_utilization
from ragas import evaluate

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[context_utilization])
score.to_pandas()
```

## Calculation 

Let's examine how context utilization was calculated using the low context utilization example:

**Step 1**: For each chunk in retrieved context, check if it is relevant or not relevant to arrive at the answer for the given question.

**Step 2**: Calculate precision@k for each chunk in the context.

```{math}
\text{Precision@1} = {\text{0} \over \text{1}} = 0
````

```{math}
\text{Precision@2} = {\text{1} \over \text{2}} = 0.5
````

**Step 3**: Calculate the mean of precision@k to arrive at the final context utilization score.

```{math}
 \text{Context Utilization} = {\text{(0+0.5)} \over \text{1}} = 0.5
```
