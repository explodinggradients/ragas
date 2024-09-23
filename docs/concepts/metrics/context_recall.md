# Context Recall

Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved. It focuses on not missing important results. Higher recall means fewer relevant documents were left out.
In short, recall is about not missing anything important. Since it is about not missing anything, calculating context recall always requires a reference to compare against.



## LLM Based Context Recall

Computed using `user_input`, `reference` and the  `retrieved_contexts`, and the values range between 0 and 1, with higher values indicating better performance. This metric uses `reference` as a proxy to `reference_contexts` which also makes it easier to use as annotating reference contexts can be very time consuming. To estimate context recall from the `reference`, the reference is broken down into claims each claim in the `reference` answer is analyzed to determine whether it can be attributed to the retrieved context or not. In an ideal scenario, all claims in the reference answer should be attributable to the retrieved context.


The formula for calculating context recall is as follows:

```{math}
\text{context recall} = {|\text{GT claims that can be attributed to context}| \over |\text{Number of claims in GT}|}
```

## Example
    
```{code-block} python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMContextRecall

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["Paris is the capital of France."], 
)

context_recall = LLMContextRecall()
await context_recall.single_turn_ascore(sample)

```

## Non LLM Based Context Recall

Computed using `retrieved_contexts` and `reference_contexts`, and the values range between 0 and 1, with higher values indicating better performance. This metrics uses non llm string comparison metrics to identify if a retrieved context is relevant or not. You can use any non LLM based metrics as distance measure to identify if a retrieved context is relevant or not.

The formula for calculating context recall is as follows:

```{math}
\text{context recall} = {|\text{Number of relevant contexts retrieved}| \over |\text{Total number of reference contexts}|}
```

## Example
    
```{code-block} python


from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

sample = SingleTurnSample(
    retrieved_contexts=["Paris is the capital of France."], 
    reference_contexts=["Paris is the capital of France.", "The Eiffel Tower is one of the most famous landmarks in Paris."]
)

context_recall = NonLLMContextRecall()
await context_recall.single_turn_ascore(sample)


```