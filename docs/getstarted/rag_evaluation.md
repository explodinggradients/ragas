## Run ragas metrics for evaluating RAG

In this tutorial, we will take a sample [test dataset](https://huggingface.co/datasets/explodinggradients/amnesty_qa), select a few of the [available metrics](/docs/concepts/metrics/index.md#list-of-available-metrics) that Ragas offers, and evaluate a simple RAG pipeline. 

### Working with Data

The dataset used here is from [Amnesty QA RAG](https://huggingface.co/datasets/explodinggradients/amnesty_qa) that contains the necessary data points we need for this tutorial. Here I am loading it from huggingface hub, but you may use file from any source. 

Converting data to ragas [evaluation dataset]()

```python
from ragas import EvaluationDataset, SingleTurnSample

samples = []
for row in dataset['eval']:
    sample = SingleTurnSample(
        user_input=row['user_input'],
        reference=row['reference'],
        response=row['response'],
        retrieved_contexts=row['retrieved_contexts']
    )
    samples.append(sample)
eval_dataset = EvaluationDataset(samples=samples)
```


### Selecting required metrics
Ragas offers a [wide variety of metrics](/docs/concepts/metrics/index.md#list-of-available-metrics) that one can select from to evaluate LLM applications. You can also build your own metrics on top of ragas. For this tutorial, we will select a few metrics that are commonly used to evaluate single turn RAG systems.

```python
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
```

Since all of the metrics we have chosen are LLM-based metrics, we need to choose the evaluator LLMs we want to use for evaluation.

### Choosing evaluator LLM

=== "OpenAI"
    This guide utilizes OpenAI for running some metrics, so ensure you have your OpenAI key ready and available in your environment.
    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "your-openai-key"
    ```
    Wrapp the LLMs in `LangchainLLMWrapper`
    ```python
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    ```


=== "AWS Bedrock"
    First you have to set your AWS credentials and configurations
    ```python
    config = {
        "credentials_profile_name": "your-profile-name",  # E.g "default"
        "region_name": "your-region-name",  # E.g. "us-east-1"
        "model_id": "your-model-id",  # E.g "anthropic.claude-v2"
        "model_kwargs": {"temperature": 0.4},
    }
    ```
    define you LLMs
    ```python
    from langchain_aws.chat_models import BedrockChat
    from ragas.llms import LangchainLLMWrapper
    evaluator_llm = LangchainLLMWrapper(BedrockChat(
        credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        endpoint_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
        model_id=config["model_id"],
        model_kwargs=config["model_kwargs"],
    ))
    ```

### Running Evaluation

```python
metrics = [LLMContextRecall(), FactualCorrectness(), Faithfulness()]
results = evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm,)
```

### Exporting and analyzing results

```python
df = result.to_pandas()
df.head()
```

