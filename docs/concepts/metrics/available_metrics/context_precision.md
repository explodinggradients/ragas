# Context Precision
Context Precision is a metric that evaluates the retriever’s ability to rank relevant chunks higher than irrelevant ones for a given query in the retrieved context. Specifically, it assesses the degree to which relevant chunks in the retrieved context are placed at the top of the ranking.

It is calculated as the mean of the precision@k for each chunk in the context. Precision@k is the ratio of the number of relevant chunks at rank k to the total number of chunks at rank k.

$$
\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant items in the top } K \text{ results}}
$$

$$
\text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})}
$$

Where $K$ is the total number of chunks in `retrieved_contexts` and $v_k \in \{0, 1\}$ is the relevance indicator at rank $k$.

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

Note that even if an irrelevant chunk is present at the second position in the array, context precision remains the same.  

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris.", "The Brandenburg Gate is located in Berlin."], 
)


await context_precision.single_turn_ascore(sample)
```
Output
```
0.9999999999
```

However, if this irrelevant chunk is placed at the first position, context precision reduces. 

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Brandenburg Gate is located in Berlin.", "The Eiffel Tower is located in Paris." ], 
)


await context_precision.single_turn_ascore(sample)
```
Output
```
0.49999999995
```

## LLM Based Context Precision

The following metrics uses LLM to identify if a retrieved context is relevant or not.

### Context Precision without reference

The `LLMContextPrecisionWithoutReference` metric can be used without the availability of a reference answer. To estimate if the retrieved contexts are relevant, this method uses the LLM to compare each chunk in `retrieved_contexts` with the `response`.

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

The `LLMContextPrecisionWithReference` metric can be used when you have both retrieved contexts and also a reference response associated with a `user_input`. To estimate if the retrieved contexts are relevant, this method uses the LLM to compare each chunk in `retrieved_contexts` with the `reference`.

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

This metric uses non-LLM-based methods (such as [Levenshtein distance measure](https://en.wikipedia.org/wiki/Levenshtein_distance)) to determine whether a retrieved context is relevant.

### Context Precision with reference contexts

The `NonLLMContextPrecisionWithReference` metric is designed for scenarios where both retrieved contexts and reference contexts are available for a `user_input`. To determine if a retrieved context is relevant, this method compares each retrieved context or chunk in `retrieved_contexts` with every context in `reference_contexts` using a non-LLM-based similarity measure.

Note that this metric would need the rapidfuzz package to be installed: `pip install rapidfuzz`. 

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
