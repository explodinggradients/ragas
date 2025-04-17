## Context Entities Recall

Context Entities Recall assesses how good the RAG system’s retriever is in fetching the entities in the reference answer. It is computed as the ratio of number of common entities between reference and the retrieved context to the total number of entities in the reference 

This metric is useful in fact-based use cases like tourism help desk, historical QA, etc. This metric can help evaluate the retrieval mechanism for entities, based on comparison with entities present in `reference`, because in cases where entities matter, we need the `retrieved_contexts` which cover them.

To compute this metric, we use two sets:  

- **$RE$**: The set of entities in the reference.  
- **$RCE$**: The set of entities in the retrieved contexts.  

We calculate the number of entities common to both sets ($RCE \cap RE$) and divide it by the total number of entities in the reference ($RE$). The formula is:  

$$
\text{Context Entity Recall} = \frac{\text{Number of common entities between $RCE$ and $RE$}}{\text{Total number of entities in $RE$}}
$$

**Metric Score Range**

- **Score Range**: 0 to 1
- **Interpretation**: A high score happens when most or all of the entities in the reference are present in the retrieved context while a low score happens when few of the entities in the reference are found in the retrieved context.

**When perfect and worst score happens**

- **When perfect score happens**: A perfect score of 1 is achieved when all entities in the reference are present in the retrieved context (100% recall of entities).
- **When worst score happens**: A score of 0 occurs when none of the entities in the reference are found in the retrieved context (0% recall of entities).


### Example

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import ContextEntityRecall
import asyncio

# Set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize the metric
evaluator_llm = LangchainLLMWrapper(llm)
context_entity_recall = ContextEntityRecall(llm=evaluator_llm)

# Define the test case
query = "What is the capital city of France?"
reference = "The capital city of France is Paris."
response = "Paris is the capital of France."
context = "France is a country in Europe with a rich history and culture."

sample = SingleTurnSample(
    user_input=query,
    reference=reference,
    response=response,
    retrieved_contexts=[context],
)

# Compute the metric
score = asyncio.run(context_entity_recall.single_turn_ascore(sample))

# Output the result
print(f"Context Entities Recall Score: {score}")
```
Output
```
Context Entities Recall Score: 0.4999999975
```

### How this metric is computed

1. **Extract Entities from Reference**: Identify and list all distinct entities (e.g., names, locations, dates) in the reference answer.
2. **Extract Entities from Retrieved Context**: Identify and list all distinct entities in the retrieved context provided by the RAG system.
3. **Find Common Entities**: Determine the intersection of the two sets, i.e., the entities that appear in both the reference and the retrieved context.
4. **Compute score**: Divide the number of common entities by the total number of entities in the reference to compute the recall score.

!!! example

    **reference**: The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in the Indian city of Agra. It was commissioned in 1631 by the Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal.

    **High entity recall context**: The Taj Mahal is a symbol of love and architectural marvel located in Agra, India. It was built by the Mughal emperor Shah Jahan in memory of his beloved wife, Mumtaz Mahal. The structure is renowned for its intricate marble work and beautiful gardens surrounding it.

    **Low entity recall context**: The Taj Mahal is an iconic monument in India. It is a UNESCO World Heritage Site and attracts millions of visitors annually. The intricate carvings and stunning architecture make it a must-visit destination.

Let us consider the refrence and the retrieved contexts given above.

- **Step-1**: Find entities present in the refrence.
    - Entities in ground truth (RE) - ['Taj Mahal', 'Yamuna', 'Agra', '1631', 'Shah Jahan', 'Mumtaz Mahal']
- **Step-2**: Find entities present in the retrieved contexts.
    - Entities in context (RCE1) - ['Taj Mahal', 'Agra', 'Shah Jahan', 'Mumtaz Mahal', 'India']
    - Entities in context (RCE2) - ['Taj Mahal', 'UNESCO', 'India']
- **Step-3**: Use the formula given above to calculate entity-recall
    
    $$
    \text{context entity recall 1} = \frac{| RCE1 \cap RE |}{| RE |}
                                 = 4/6
                                 = 0.666
    $$

    $$
    \text{context entity recall 2} = \frac{| RCE2 \cap RE |}{| RE |}
                                 = 1/6
    $$

    We can see that the first context had a high entity recall, because it has a better entity coverage given the refrence. If these two retrieved contexts were fetched by two retrieval mechanisms on same set of documents, we could say that the first mechanism was better than the other in use-cases where entities are of importance.

