

# Noise Sensitivity

Noise sensitivity measures how often a system makes errors by providing incorrect responses when utilizing either relevant or irrelevant retrieved documents. The score ranges from 0 to 1, with lower values indicating better performance. Noise sensitivity is computed using the question, ground truth, answer, and the retrieved context.

To estimate noise sensitivity, each claim in the generated answer is examined to determine whether it is correct based on the ground truth and whether it can be attributed to the relevant (or irrelevant) retrieved context. Ideally, all claims in the answer should be supported by the relevant retrieved context.


```{math}
\text{noise sensitivity (relevant)} = {|\text{Number of incorrect claims in answer}| \over |\text{Number of claims in the Answer}|}
```

```{Hint}

Question: What is the Life Insurance Corporation of India (LIC) known for?

Ground truth: The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.

Relevant Retrieval: 
    - The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.
    - LIC is the largest insurance company in India, with a vast network of policyholders and a significant role in the financial sector.
    - As the largest institutional investor in India, LIC manages a substantial life fund, contributing to the financial stability of the country.
       
Irrelevant Retrieval: 
    - The Indian economy is one of the fastest-growing major economies in the world, thanks to the secors like finance, technology, manufacturing etc.
```


## Example

```{code-block} python
:caption: Noise Sensitivity
from datasets import Dataset 
from ragas.metrics import noise_sensitivity_relevant, noise_sensitivity_irrelevant
from ragas import evaluate

data_sample = {
    "question": ["What is the Life Insurance Corporation of India (LIC) known for?"],
    "ground_truth": ["The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments."],
    "answer": ["The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributs to the financial stability of the country."],
    "contexts": [[
        "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
        "LIC is the largest insurance company in India, with a vast network of policyholders and a huge investments.",
        "As the largest institutional investor in India, LIC manages a substantial funds, contributing to the financial stability of the country.",
        "The Indian economy is one of the fastest-growing major economies in the world, thanks to the secors like finance, technology, manufacturing etc"
    ]]
}

dataset = Dataset.from_dict(data_sample)
metrics = [noise_sensitivity_relevant, noise_sensitivity_irrelevant]
score = evaluate(dataset,metrics=metrics)
score.to_pandas()
```

## Calculation

Let's examine how noise sensitivity in relevant context was calculated:

- **Step 1:** Identify the relevant contexts from which the ground truth can be inferred.

    - Ground Truth:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.  

    - Contexts:
        - Context 1: `The Life Insurance Corporation of India (LIC) was established in 1956` following the nationalization of the insurance industry in India.
        - Context 2: `LIC is the largest insurance company in India`, with a vast network of policyholders and a significant role in the financial sector.
        - Context 3: `As the largest institutional investor in India, LIC manages a substantial funds`, contributing to the financial stability of the country.

- **Step 2:** Verify if the claims in the generated answer can be inferred from the relevant context.

    - Answer:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributs to the financial stability of the country.

    - Contexts:
        - Context 1: The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.
        - Context 2: `LIC is the largest insurance company in India`, with a vast network of policyholders and a significant role in the financial sector.
        - Context 3: `As the largest institutional investor in India, LIC manages a substantial funds`, `contributing to the financial stability of the country`.


- **Step 3:** Identify any incorrect claims in the answer (i.e., answer statements that are not supported by the ground truth).

    - Ground Truth:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.

    - Answer:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. `LIC contributs to the financial stability of the country`.

    Explanation: The ground truth does not mention anything about LIC contributing to the financial stability of the country. Therefore, this statement in the answer is incorrect.

    Incorrect Statement: 1
    Total claims: 3

- **Step 4:** Calculate noise sensitivity using the formula:
    ```{math}
    \text{noise sensitivity} = { \text{1} \over \text{3} } = 0.333
    ``` 
This results in a noise sensitivity score of 0.333, indicating that one out of three claims in the answer was incorrect.


Credits: Noise senstivity was introduced in [RAGChecker](https://github.com/amazon-science/RAGChecker/tree/main/ragchecker)