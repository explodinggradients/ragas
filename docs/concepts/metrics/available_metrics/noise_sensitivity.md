# Noise Sensitivity

`NoiseSensitivity` measures how often a system makes errors by providing incorrect responses when utilizing either relevant or irrelevant retrieved documents. The score ranges from 0 to 1, with lower values indicating better performance. Noise sensitivity is computed using the `user_input`,  `reference`, `response`, and the `retrieved_contexts`.

To estimate noise sensitivity, each claim in the generated response is examined to determine whether it is correct based on the ground truth and whether it can be attributed to the relevant (or irrelevant) retrieved context. Ideally, all claims in the answer should be supported by the relevant retrieved context.


$$
\text{noise sensitivity (relevant)} = {|\text{Total number of incorrect claims in response}| \over |\text{Total number of claims in the response}|}
$$



## Example

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NoiseSensitivity

sample = SingleTurnSample(
    user_input="What is the Life Insurance Corporation of India (LIC) known for?",
    response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
    reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
    retrieved_contexts=[
        "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
        "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
        "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
        "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc."
    ]
)

scorer = NoiseSensitivity()
await scorer.single_turn_ascore(sample)
```

To calculate noise sensivity of irrelevant context, you can set the `focus` parameter to `irrelevant`.

```python
scorer = NoiseSensitivity(focus="irrelevant")
await scorer.single_turn_ascore(sample)
```

## How It’s Calculated

!!! example
    Question: What is the Life Insurance Corporation of India (LIC) known for?

    Ground truth: The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.

    Relevant Retrieval: 
        - The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.
        - LIC is the largest insurance company in India, with a vast network of policyholders and a significant role in the financial sector.
        - As the largest institutional investor in India, LIC manages a substantial life fund, contributing to the financial stability of the country.
        
    Irrelevant Retrieval: 
        - The Indian economy is one of the fastest-growing major economies in the world, thanks to the sectors like finance, technology, manufacturing etc.

Let's examine how noise sensitivity in relevant context was calculated:

- **Step 1:** Identify the relevant contexts from which the ground truth can be inferred.

    - Ground Truth:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.  

    - Contexts:
        - Context 1: The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.
        - Context 2: LIC is the largest insurance company in India, with a vast network of policyholders and a significant role in the financial sector.
        - Context 3: As the largest institutional investor in India, LIC manages a substantial funds`, contributing to the financial stability of the country.

- **Step 2:** Verify if the claims in the generated answer can be inferred from the relevant context.

    - Answer:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributs to the financial stability of the country.

    - Contexts:
        - Context 1: The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.
        - Context 2: LIC is the largest insurance company in India, with a vast network of policyholders and a significant role in the financial sector.
        - Context 3: As the largest institutional investor in India, LIC manages a substantial funds, contributing to the financial stability of the country.


- **Step 3:** Identify any incorrect claims in the answer (i.e., answer statements that are not supported by the ground truth).

    - Ground Truth:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.

    - Answer:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributs to the financial stability of the country.

    Explanation: The ground truth does not mention anything about LIC contributing to the financial stability of the country. Therefore, this statement in the answer is incorrect.

    Incorrect Statement: 1
    Total claims: 3

- **Step 4:** Calculate noise sensitivity using the formula:

    $$
    \text{noise sensitivity} = { \text{1} \over \text{3} } = 0.333
    $$ 

This results in a noise sensitivity score of 0.333, indicating that one out of three claims in the answer was incorrect.


Credits: Noise senstivity was introduced in [RAGChecker](https://github.com/amazon-science/RAGChecker/tree/main/ragchecker)