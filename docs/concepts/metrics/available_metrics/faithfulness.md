## Faithfulness

The **Faithfulness metric** measures how factually consistent a generated response is with the retrieved context. It evaluates whether the claims  in the response can be inferred from the retrieved context.  It is computed as the ratio of number of claims in the response supported by retrieved context to the total number of claims in the response. 

The Faithfulness metric is computed based on:

- **Generated Response**: The answer provided by the RAG system.
- **Retrieved Context**: The text chunks retrieved by the RAG system.

The Faithfulness metric score is calculated as:

$$
\text{Faithfulness Score} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}
$$

**Metric Score Range**

- **Range**: 0 to 1
- **Interpretation**: A high score indicates greater factual consistency with the retrieved context, while a low score indicates discrepancies or unsupported claims.

**When Perfect and Worst Score Happens**

- **Perfect Score (1)**: Occurs when all claims in the generated response are fully supported by the retrieved context, with no contradictions.
- **Worst Score (0)**: Occurs when none of the claims in the generated response can be inferred from the retrieved context, indicating complete factual inconsistency.


### Example

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
import asyncio

# Set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize the metric
evaluator_llm = LangchainLLMWrapper(llm)
faithfulness = Faithfulness(llm=evaluator_llm)

# Define the test sample
query = "What are some tips for maintaining a healthy diet?"
response = "Eating fruits and vegetables daily, drinking enough water, and avoiding processed foods can improve your diet."
context = [
    "A healthy diet includes regular consumption of fruits and vegetables.",
    "Staying hydrated by drinking sufficient water is essential for good health.",
    "Processed foods should be limited to maintain a balanced diet."
]

sample = SingleTurnSample(
    user_input = query,
    response = response,
    retrieved_contexts = context,
)

# Compute the metric
score = asyncio.run(faithfulness.single_turn_ascore(sample))

# Display the score
print(f"Faithfulness Score: {score}")
```
Output
```
Faithfulness Score: 1.0
```

## Faithfullness with HHEM-2.1-Open

[Vectara's HHEM-2.1-Open](https://vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model/) is a classifier model (T5) that is trained to detect hallucinations from LLM generated text. This model can be used in the second step of calculating faithfulness, i.e. when claims are cross-checked with the given context to determine if it can be inferred from the context. The model is free, small, and open-source, making it very efficient in production use cases. To use the model to calculate faithfulness, you can use the following code snippet:

```python

# Install Hugging Face transformers library: pip install transformers
# HHEM-2.1-Open model is downloaded and run using transformers library.

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import FaithfulnesswithHHEM
import asyncio

# Set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize the metric
evaluator_llm = LangchainLLMWrapper(llm)
faithfulness = FaithfulnesswithHHEM(llm=evaluator_llm)

# Define the test sample
query = "What are some tips for maintaining a healthy diet?"
response = "Eating fruits and vegetables daily, drinking enough water, and avoiding processed foods can improve your diet."
context = [
    "A healthy diet includes regular consumption of fruits and vegetables.",
    "Staying hydrated by drinking sufficient water is essential for good health.",
    "Processed foods should be limited to maintain a balanced diet."
]

sample = SingleTurnSample(
    user_input = query,
    response = response,
    retrieved_contexts = context,
)

# Compute the metric
score = asyncio.run(faithfulness.single_turn_ascore(sample))

# Display the score
print(f"Faithfulness Score: {score}")

```

You can load the model onto a specified device by setting the `device` argument and adjust the batch size for inference using the `batch_size` parameter. By default, the model is loaded on the CPU with a batch size of 10

```python

my_device = "cuda:0"
my_batch_size = 10

faithfulness = FaithfulnesswithHHEM(device=my_device, batch_size=my_batch_size)
score = asyncio.run(faithfulness.single_turn_ascore(sample))
```

### How this metric is computed 

1. **Extract Claims**: Use an LLM  to break down the generated response into individual factual claims.
2. **Verify Claims Against Context**: For each claim, check if it can be supported or inferred from the retrieved context.
    - A claim is "supported" if it aligns with the context without contradiction.
    - A claim is "unsupported" if it contradicts the context or has no basis in it.
3. **Count Supported Claims**: Count the number of claims that are supported by the context.
4. **Calculate Score**: Compute faithfulness score as the ratio of number of  supported claims and the total number of claims.

!!! example

    **Question**: "When was the first Super Bowl held?"

    **Retrieved Context**: "The First AFL–NFL World Championship Game, later known as Super Bowl I, was played on January 15, 1967, at the Los Angeles Memorial Coliseum."

    **Generated Response**: "The first Super Bowl was held on January 15, 1967, in Florida."

**Step-by-Step Computation:**

1. **Extract Claims**:
    - Claim 1: "The first Super Bowl was held on January 15, 1967."
    - Claim 2: "The first Super Bowl was held in Florida."
    - Total claims = 2.
2. **Verify Claims Against Context**:
    - Claim 1: "The first Super Bowl was held on January 15, 1967."
        - Supported by context: "played on January 15, 1967."
        - Verdict: Yes (supported).
    - Claim 2: "The first Super Bowl was held in Florida."
        - Contradicted by context: "at the Los Angeles Memorial Coliseum" (in California, not Florida).
        - Verdict: No (unsupported).
3. **Count Supported Claims**:
    - Number of supported claims = 1.
4. **Calculate Score**:

$$
\text{Faithfulness Score} = \frac{\text{Number of supported claims}}{\text{Total claims}} = \frac{1}{2} = 0.5
$$

The response is only partially faithful. The date is correct, but the location contradicts the retrieved context, lowering the score.

