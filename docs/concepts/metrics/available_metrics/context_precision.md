# Context Precision
The Context Precision metric is used to evaluate the retrieval component of RAG systems. This metric evaluates the retriever’s ability to rank relevant chunks higher than irrelevant ones for a given query in the retrieved context.  Specifically, it assesses whether the relevant chunks in the retrieved context are prioritized at the top of the ranking. This is important because  if relevant chunks are ranked lower, the model may prioritize irrelevant or less useful information, degrading the quality of the generated response.  

It is calculated as the mean of the precision@k for each chunk in the context. Precision@k is the ratio of the number of relevant chunks at rank k to the total number of chunks at rank k.

$$
\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant items in the top } K \text{ results}}
$$

$$
\text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})}
$$

Where $K$ is the total number of chunks in `retrieved_contexts` and $v_k \in \{0, 1\}$ is the relevance indicator at rank $k$.

**Metric score range**

- **Score range**: 0 to 1
- **Interpretation**: A high score  occurs when  relevant chunks are ranked at the top of the list. A low score occurs when  relevant chunks are buried lower in the ranking, overshadowed by irrelevant ones.

**When perfect and worst score happens**

- **When perfect score happens**: A perfect score of 1 is achieved when all retrieved chunks are relevant and they are ranked in an order that perfectly prioritizes them (i.e., no irrelevant chunks precede relevant ones).
- **When worst score happens**: A score of 0 is obtained when none of the retrieved chunks are relevant, regardless of their ranking.


## LLM Based Context Precision

The following metrics uses LLM to identify if a retrieved context is relevant or not.

### Context Precision without reference

`LLMContextPrecisionWithoutReference` metric can be used when you have only retrieved contexts associated with a `user_input`. To estimate if a retrieved contexts is relevant or not this method uses the LLM to compare each of the retrieved context or chunk present in `retrieved_contexts` with `response`.

#### Example
    
```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
import asyncio

# Set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize the metric
evaluator_llm = LangchainLLMWrapper(llm)
context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

# Defin the test sample
query = "Where is the Eiffel Tower located?"
response = "The Eiffel Tower is located in Paris."
context = ["The Eiffel Tower is located in Paris."]

sample = SingleTurnSample(
    user_input=query,
    response=response,
    retrieved_contexts=context, 
)

# Compute the metric score
score = asyncio.run(context_precision.single_turn_ascore(sample))

# Display the metric score
print(f"Context Precision Score: {score}")
```
Output
```
Context Precision Score: 0.9999999999
```

### Context Precision with reference

`LLMContextPrecisionWithReference` metric is can be used when you have both retrieved contexts and also reference answer associated with a `user_input`. To estimate if a retrieved contexts is relevant or not this method uses the LLM to compare each of the retrieved context or chunk present in `retrieved_contexts` with `reference`. 

#### Example
    
```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference
import asyncio

# Set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize the metric
evaluator_llm = LangchainLLMWrapper(llm)
context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)

# Define the test sample
query = "Where is the Eiffel Tower located?"
reference = "The Eiffel Tower is located in Paris."
context = ["The Eiffel Tower is located in Paris."]

sample = SingleTurnSample(
    user_input=query,
    reference=reference,
    retrieved_contexts=context, 
)

# Compute the metric score
score = asyncio.run(context_precision.single_turn_ascore(sample))

# Display the metric score
print(f"Context Precision Score: {score}")
```
Output
```
Context Precision Score: 0.9999999999
```

#### How this metric is computed 

1. **Identify Relevance**: For each chunk in the retrieved contexts, determine if it is relevant to the reference answer. 
2. **Assign Relevance Indicators**: Assign $v_k = 1$  for relevant chunks and  $v_k = 0$ for irrelevant ones.
3. **Calculate Precision@k**: For each rank ( k ) (from 1 to  K), compute the precision as the number of relevant chunks up to that rank divided by the total number of chunks up to that rank.
4. **Compute Weighted Sum**: Multiply each $\text{Precision@k}$  by its corresponding $v_k$  and sum these values across all ranks.
5. **Compute Score**: Divide the sum by the total number of relevant chunks in the top ( K ) chunks.

---

**Example**

- **Question**: "What is the largest desert in the world?"
- **Ground Truth**: "The largest desert in the world is the Antarctic Desert, which spans about 14 million square kilometers."
- **Retrieved Contexts**:
    1. "The Antarctic Desert is the largest desert by area, covering 14 million square kilometers."
    2. "The Sahara Desert is a large desert in Africa."
    3. "Deserts are dry regions with little rainfall."

**Step by step computation**

1. **Relevance Check**:
    - Chunk 1: Relevant  – directly answers the query by identifying the largest desert.
    - Chunk 2: Irrelevant – mentions a desert but not the largest one.
    - Chunk 3: Irrelevant  – too general, doesn’t specify the largest desert.
2. **Assign Relevance Indicators**:
    -  Chunk 1: Relevant ( $v_1 = 1$) 
    - Chunk 2: Irrelevant ( $v_2 = 0$) 
    -  Chunk 3: Irrelevant ($v_3 = 0$) 

1. **Compute Precision@k**:
    - At $k = 1$ : 1 relevant / 1 total = 1.0
    - At  $k = 2$: 1 relevant / 2 total = 0.5
    - At $k = 3$ : 1 relevant / 3 total = 0.33
2. **Compute Weighted Sum**:
    - $(1.0 \times 1) + (0.5 \times 0) + (0.33 \times 0) = 1.0 + 0 + 0 = 1.0$
3. **Compute Score**: Total relevant items in ground truth = 1 (only the Antarctic Desert is correct).
    - $\text{Context Precision@3} = \frac{1.0}{1} = 1.0$

Context Precision = 1.0  because the only relevant chunk is ranked at the top.

Now, if the order was reversed (irrelevant chunks ranked higher), the score would drop, reflecting poorer precision due to lower ranking of the relevant chunk.

## Non LLM Based Context Precision

This metric uses traditional methods to determine whether a retrieved context is relevant. It relies on non-LLM-based metrics as a distance measure to evaluate the relevance of retrieved contexts.

### Context Precision with reference contexts

The `NonLLMContextPrecisionWithReference` metric is designed for scenarios where both retrieved contexts and reference contexts are available for a `user_input`. To determine if a retrieved context is relevant, this method compares each retrieved context or chunk in `retrieved_context`s with every context in `reference_contexts` using a non-LLM-based similarity measure.

#### Example
    
```python
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference

# Initialize the metric
context_precision = NonLLMContextPrecisionWithReference()

# Define the test sample
sample = SingleTurnSample(
    retrieved_contexts=["The Eiffel Tower is located in Paris."], 
    reference_contexts=["Paris is the capital of France.", "The Eiffel Tower is one of the most famous landmarks in Paris."]
)

# Compute the metric score
score = asyncio.run(context_precision.single_turn_ascore(sample))

# Display the metric score
print(f"Context Precision Score: {score}")
```
Output
```
Context Precision Score: 0.9999999999
```