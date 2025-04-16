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

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."], 
)


await context_precision.single_turn_ascore(sample)
```
Output
```
0.9999999999
```

### Context Precision with reference

`LLMContextPrecisionWithReference` metric is can be used when you have both retrieved contexts and also reference answer associated with a `user_input`. To estimate if a retrieved contexts is relevant or not this method uses the LLM to compare each of the retrieved context or chunk present in `retrieved_contexts` with `reference`. 

#### Example
    
```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference

context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."], 
)

await context_precision.single_turn_ascore(sample)
```
Output
```
0.9999999999
```

## Non LLM Based Context Precision

This metric uses traditional methods to determine whether a retrieved context is relevant. It relies on non-LLM-based metrics as a distance measure to evaluate the relevance of retrieved contexts.

### Context Precision with reference contexts

The `NonLLMContextPrecisionWithReference` metric is designed for scenarios where both retrieved contexts and reference contexts are available for a `user_input`. To determine if a retrieved context is relevant, this method compares each retrieved context or chunk in `retrieved_context`s with every context in `reference_contexts` using a non-LLM-based similarity measure.

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
Output
```
0.9999999999
```

## ID Based Context Precision

IDBasedContextPrecision provides a direct and efficient way to measure precision by comparing the IDs of retrieved contexts with reference context IDs. This metric is particularly useful when you have a unique ID system for your documents and want to evaluate retrieval performance without comparing the actual content.

The metric computes precision using retrieved_context_ids and reference_context_ids, with values ranging between 0 and 1. Higher values indicate better performance. It works with both string and integer IDs.

The formula for calculating ID-based context precision is as follows:

$$ \text{ID-Based Context Precision} = \frac{\text{Number of retrieved context IDs found in reference context IDs}}{\text{Total number of retrieved context IDs}} $$

### Example

```python
from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision

sample = SingleTurnSample(
    retrieved_context_ids=["doc_1", "doc_2", "doc_3", "doc_4"], 
    reference_context_ids=["doc_1", "doc_4", "doc_5", "doc_6"]
)

id_precision = IDBasedContextPrecision()
await id_precision.single_turn_ascore(sample)

```

Output
```
0.5
```

In this example, out of the 4 retrieved context IDs, only 2 ("doc_1" and "doc_4") are found in the reference context IDs, resulting in a precision score of 0.5 or 50%.