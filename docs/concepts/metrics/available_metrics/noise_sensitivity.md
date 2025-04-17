# Noise Sensitivity

Noise Sensitivity measures the sensitivity of RAG system's generator to the noise (incorrect claims) in the 
retrieved context. This metric is computed in two modes namely: Relevant Noise Sensitivity and Irrelevant Noise Sensitivity.

## Relevant Noise Sensitivity

Relevant noise sensitivity metric measures generator’s sensitivity to noise (incorrect claims) in relevant retrieved context. It is computed as ratio of incorrect claims in the response that are entailed by relevant retrieved chunks to the total number of response claims. A relevant chunk is a piece of retrieved context that contains at least one claim from the reference. 

This metric is computed based on 

- **Reference**: The correct answer to the user query.
- **Retrieved Relevant Chunks**: Chunks that contain at least one claim from the reference.
- **Response**: The generated output from the RAG system.

The Relevant Noise Sensitivity (RNS) is computed using the formula:

$$
\text{Relevant Noise Sensitivity} = \frac{\text{Number of incorrect claims in the model response entailed by relevant chunks}}{\text{Total number of claims in the response}}
$$

---

**Metric Score Range**

- **Range**: 0 to 1
- **Interpretation**: Higher score means the generator is highly sensitive to noise while lower score means the generator is more robust to noise.

---

**When Perfect and Worst Score Happens**

- **Perfect Score (0)**: Occurs when the model response contains no incorrect claims, or when any incorrect claims present are not supported by any relevant retrieved chunks. This indicates the generator successfully filters out noise from relevant context.
- **Worst Score (1)**: Occurs when every incorrect claim in the model response is directly supported by a relevant retrieved chunk. This suggests that the generator is entirely misled by noise in the relevant context.

### Example

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import NoiseSensitivity
import asyncio

# Set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize the metric
evaluator_llm = LangchainLLMWrapper(llm)
relevant_noise_sensitivity = NoiseSensitivity(llm=evaluator_llm, mode="relevant")

# Define the test sample
query = "Who painted the Mona Lisa and in what century was it painted?"
response = "Leonardo da Vinci painted the Mona Lisa, and it was painted in the 15th century."
reference = "Leonardo da Vinci painted the Mona Lisa in the 16th century."
context = [
    "The Mona Lisa is a famous portrait painted by Leonardo da Vinci. It is believed to have been started in the early 1500s. Some art historians date its completion to around 1519, placing it firmly in the 15th century according to certain periodizations."
]

sample = SingleTurnSample(
    user_input = query,
    response = response,
    reference = reference,
    retrieved_contexts = context,
)

# Compute the metric
score = asyncio.run(relevant_noise_sensitivity.single_turn_ascore(sample))

# Display the score
print(f"Relevant Noise Sensitivity Score: {score}")

```

Output
```
Relevant Noise Sensitivity Score: 0.5
```

### How this metric is computed

1. **Extract Reference Claims**: Break down the reference into individual factual claims.
2. **Identify Relevant Chunks**: From the retrieved context, identify relevant chunks i.e., chunks with contain at least one claim from the reference. 
3. **Extract Response Claims**: Break the response into individual claims. 
4. **Classify Claims**: Compare response claims to the reference to label them as correct or incorrect.
5. **Check Entailment**: For each incorrect claim in the response, determine if it is entailed (supported or implied) by a relevant chunk. 
6. **Compute the Metric**:
    - Numerator: Count incorrect claims entailed by relevant chunks.
    - Denominator: Count total claims in the response.
    - Divide to get the relevant noise sensitivity score.

!!! example

    **User Query**: "Who painted the Mona Lisa and in what century was it painted?"

    **Reference**: "Leonardo da Vinci painted the Mona Lisa in the 16th century."

    **Retrieved Context**: "The Mona Lisa is a famous portrait painted by Leonardo da Vinci. It is believed to have been started in the early 1500s. Some art historians date its completion to around 1519, placing it firmly in the 15th century according to certain periodizations."

    **Response**: "Leonardo da Vinci painted the Mona Lisa, and it was painted in the 15th century."

**Step by Step Computation**

1. **Extract Reference Claims**:
    - Claim 1: "Leonardo da Vinci painted the Mona Lisa" .
    - Claim 2: "It was painted in the 16th century" .
2. **Relevant Chunk Identification**: The retrieved chunk is relevant because it contains "Leonardo da Vinci painted the Mona Lisa," matching the ground truth.
3. **Extract Response Claims**:
    - Claim 1: "Leonardo da Vinci painted the Mona Lisa" .
    - Claim 2: "It was painted in the 15th century".
4. **Classify Claims**:
    - Correct: 1 claim ("Leonardo da Vinci painted the Mona Lisa").
    - Incorrect: 1 claim ("It was painted in the 15th century").
5. **Check Entailment**:
    - Incorrect claim ("15th century") is entailed by the relevant chunk’s statement: "...placing it firmly in the 15th century according to certain periodizations."
6. **Compute the Metric**:
    - Numerator: 1 (incorrect claim entailed by a relevant chunk).
    - Denominator: 2 (total claims in the response).
    - $score = \frac{1}{2} = 0.5$

The Relevant Noise Sensitivity is **0.5**, indicating that half of the claims in the model response are incorrect and stem from noise in a relevant retrieved chunk. This reflects moderate sensitivity to noise.


## Irrelevant Noise Sensitivity

Irrelevant noise sensitivity metric evaluates generator’s sensitivity to noise (incorrect claims) in irrelevant retrieved context. It is computed as ratio of incorrect claims in the response that are entailed by irrelevant retrieved chunks to the total number of response claims. An irrelevant chunk is a piece of retrieved context that contains no claim from the reference. 

**This metric is computed based on**

- **Reference**: The correct answer to the user query.
- **Response**: The generated output from the RAG system.
- **Irrelevant Retrieved Chunks**: Chunks that contain no ground truth claims.

This metric is computed using the formula:

$$
\text{Irrelevant Noise Sensitivity} = \frac{\text{Number of incorrect claims in the response entailed by irrelevant chunks}}{\text{Total number of claims in the response}}
$$

---

**Metric Score Range**

Metrics Score Range: 0 to 1  

Interpretation:  High Score indicates the model is highly sensitive to irrelevant noise, meaning most incorrect claims in the response are derived from irrelevant retrieved chunks. Low Score suggests the model is robust against irrelevant noise, with few or no incorrect claims supported by irrelevant chunks.

**When Perfect and Worst Scores Happen**

- **Perfect Score (0)**: The response either has no incorrect claims, or any incorrect claims are not derived from irrelevant chunks (e.g., they might come from the model’s own errors or relevant chunks).
- **Worst Score (1)**: Every incorrect claim in the response is traceable to irrelevant retrieved chunks, indicating the model is highly susceptible to irrelevant noise.

### Example

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import NoiseSensitivity
import asyncio

# Set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize the metric
evaluator_llm = LangchainLLMWrapper(llm)
irrelevant_noise_sensitivity = NoiseSensitivity(llm=evaluator_llm, mode="irrelevant")

# Define the test sample
query = "Who wrote the novel 'Pride and Prejudice'?"
response = "Charlotte Brontë wrote 'Pride and Prejudice,' and she is famous for 'Jane Eyre.'"
reference = "Jane Austen wrote 'Pride and Prejudice.'"
context = [
    "Jane Austen published 'Pride and Prejudice' in 1813, a classic romance novel.",
    "Charlotte Brontë, a renowned author, is best known for her novel 'Jane Eyre,' published in 1847."
]

sample = SingleTurnSample(
    user_input = query,
    response = response,
    reference = reference,
    retrieved_contexts = context,
)

# Compute the metric
score = asyncio.run(irrelevant_noise_sensitivity.single_turn_ascore(sample))

# Display the score
print(f"Irrelevant Noise Sensitivity Score: {score}")
```

Output
```
Irrelevant Noise Sensitivity Score: 0.5
```

### How This Metric Is Computed

1. **Extract Reference Claims**: Break down the reference into individual claims.
2. **Extract Response Claims**: Break down the response into individual claims.
3. **Identify Irrelevant Chunks**: Determine which chunks are irrelevant (do not entail any reference claim).
4. **Identify Incorrect Claims**: Compare response claims to the reference to label them as correct or incorrect.
5. **Check Entailment with Irrelevant Chunks**: For each incorrect response claim, determine if it is supported by any irrelevant chunk.
6. **Compute the Metric**:
    - Numerator: Count incorrect response claims entailed by irrelevant chunks.
    - Denominator: Count all claims in the response.
    - Divide to get the score.


!!! example

    **Query**: "Who wrote the novel 'Pride and Prejudice'?" 

    **Ground Truth Answer**: "Jane Austen wrote 'Pride and Prejudice.'"

    **Model Response**: "Charlotte Brontë wrote 'Pride and Prejudice,' and she is famous for 'Jane Eyre.'"

    **Retrieved Context**:

    - "Jane Austen published 'Pride and Prejudice' in 1813, a classic romance novel.".
    - "Charlotte Brontë, a renowned author, is best known for her novel 'Jane Eyre,' published in 1847."

**Steps**:

1. **Extract Reference Claims**: Re1 = "Jane Austen wrote 'Pride and Prejudice.'"
2. **Extract Response Claims**: Rs1 = "Charlotte Brontë wrote 'Pride and Prejudice,'",  Rs2 = "Charlotte Brontë is famous for 'Jane Eyre.'"
3. **Identify Irrelevant Chunks**:  IR1- "Charlotte Brontë, a renowned author, is best known for her novel 'Jane Eyre,' published in 1847." is irrelevant.
4. **Identify Incorrect Claims**:
    - Rs1 (incorrect).
    - Rs2 (incorrect).
5. **Entailment Check**:
    - Rs1 ("Charlotte Brontë wrote 'Pride and Prejudice'") is not entailed by IR1.
    - Rs2 ("Charlotte Brontë is famous for 'Jane Eyre'") is entailed by IR1.
6. **Compute the Metric**:
    - Numerator: Number of incorrect response claims entailed by irrelevant chunks = 1 (only Rs2).
    - Denominator: Total number of claims in the response = 2 (Rs1 and Rs2).
    - score =  $\frac{1}{2} = 0.5$

50% of the model’s claims that deviate from the ground truth are supported by irrelevant retrieved chunks. This suggests the model is moderately sensitive to irrelevant noise.


Credits: Noise senstivity was introduced in [RAGChecker](https://github.com/amazon-science/RAGChecker/tree/main/ragchecker)