# Context Precision
Context Precision is a metric that measures the proportion of relevant chunks in the `retrieved_contexts`. It is calculated as the mean of the precision@k for each chunk in the context. Precision@k is the ratio of the number of relevant chunks at rank k to the total number of chunks at rank k.

$$
\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant items in the top } K \text{ results}}
$$

$$
\text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})}
$$

Where $K$ is the total number of chunks in `retrieved_contexts` and $v_k \in \{0, 1\}$ is the relevance indicator at rank $k$.

## LLM Based Context Precision

The following metrics uses LLM to identify if a retrieved context is relevant or not.

### Context Precision without reference

`LLMContextPrecisionWithoutReference` metric can be used when you have both retrieved contexts and also reference contexts associated with a `user_input`. To estimate if a retrieved contexts is relevant or not this method uses the LLM to compare each of the retrieved context or chunk present in `retrieved_contexts` with `response`.

#### Example
    
```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference()

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."], 
)


await context_precision.single_turn_ascore(sample)
```

### Context Precision with reference

`LLMContextPrecisionWithReference` metric is can be used when you have both retrieved contexts and also reference answer associated with a `user_input`. To estimate if a retrieved contexts is relevant or not this method uses the LLM to compare each of the retrieved context or chunk present in `retrieved_contexts` with `reference`. 

#### Example
    
```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference

context_precision = LLMContextPrecisionWithReference()

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."], 
)

await context_precision.single_turn_ascore(sample)
```

## Non LLM Based Context Precision

The following metrics uses traditional methods to identify if a retrieved context is relevant or not. You can use any non LLM based metrics as distance measure to identify if a retrieved context is relevant or not.

### Context Precision with reference contexts

`NonLLMContextPrecisionWithReference` metric is can be used when you have both retrieved contexts and also reference contexts associated with a `user_input`. To estimate if a retrieved contexts is relevant or not this method uses the LLM to compare each of the retrieved context or chunk present in `retrieved_contexts` with each ones present in `reference_contexts`. 

#### Example
    
```python
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference

context_precision = NonLLMContextPrecisionWithReference()

sample = SingleTurnSample(
    retrieved_contexts=["The Eiffel Tower is located in Paris."], 
    reference_contexts=["Paris is the capital of France.", "The Eiffel Tower is one of the most famous landmarks in Paris."]
)

await context_precision.single_turn_ascore(sample)
```