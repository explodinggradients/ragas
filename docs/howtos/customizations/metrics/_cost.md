# Understand Cost and Usage of Operations

When using LLMs for evaluation and test set generation, cost will be an important factor. Ragas provides several tools to help you optimize costs, including **Batch API support** for up to 50% savings on large-scale evaluations.

## Cost Optimization Strategies

### 1. Use Batch API for Large Evaluations (50% Savings)

For non-urgent evaluation workloads, Ragas supports OpenAI's Batch API which provides 50% cost savings:

```python
from ragas.batch_evaluation import BatchEvaluator, estimate_batch_cost_savings
from ragas.metrics import Faithfulness
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# Setup batch-capable LLM
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
faithfulness = Faithfulness(llm=llm)

# Estimate cost savings
cost_info = estimate_batch_cost_savings(
    sample_count=1000,
    metrics=[faithfulness],
    regular_cost_per_1k_tokens=0.15,  # GPT-4o-mini cost
    batch_discount=0.5  # 50% savings
)

print(f"Regular cost: ${cost_info['regular_cost']}")
print(f"Batch cost: ${cost_info['batch_cost']}")  
print(f"Savings: ${cost_info['savings']} ({cost_info['savings_percentage']}%)")

# Run batch evaluation
evaluator = BatchEvaluator(metrics=[faithfulness])
results = evaluator.evaluate(samples, wait_for_completion=True)
```

Learn more about [Batch Evaluation](batch_evaluation.md).

### 2. Monitor Token Usage

## Understanding `TokenUsageParser`

By default, Ragas does not calculate the usage of tokens for `evaluate()`. This is because LangChain's LLMs do not always return information about token usage in a uniform way. So in order to get the usage data, we have to implement a `TokenUsageParser`.

A `TokenUsageParser` is function that parses the `LLMResult` or `ChatResult` from LangChain models `generate_prompt()` function and outputs `TokenUsage` which Ragas expects.

For an example here is one that will parse OpenAI by using a parser we have defined.


```python
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"
```


```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompt_values import StringPromptValue

gpt4o = ChatOpenAI(model="gpt-4o")
p = StringPromptValue(text="hai there")
llm_result = gpt4o.generate_prompt([p])

# lets import a parser for OpenAI
from ragas.cost import get_token_usage_for_openai

get_token_usage_for_openai(llm_result)
```

```py
/opt/homebrew/Caskroom/miniforge/base/envs/ragas/lib/python3.9/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
    from .autonotebook import tqdm as notebook_tqdm

TokenUsage(input_tokens=9, output_tokens=9, model='')
```


You can define your own or import parsers if they are defined. If you would like to suggest parser for LLM providers or contribute your own ones please check out this [issue](https://github.com/explodinggradients/ragas/issues/1151) 🙂.

You can use it for evaluations as so. Using example from [get started](get-started-evaluation) here.


```python
from datasets import load_dataset
from ragas import EvaluationDataset
from ragas.metrics._aspect_critic import AspectCriticWithReference

dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3")


eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])

metric = AspectCriticWithReference(
    name="answer_correctness",
    definition="is the response correct compared to reference",
)
```

```py
Repo card metadata block was not found. Setting CardData to empty.
```

```python
from ragas import evaluate
from ragas.cost import get_token_usage_for_openai

results = evaluate(
    eval_dataset[:5],
    metrics=[metric],
    llm=gpt4o,
    token_usage_parser=get_token_usage_for_openai,
)
```

```py
Evaluating: 100%|██████████| 5/5 [00:01<00:00,  2.81it/s]
```

```python
results.total_tokens()
```

```py
TokenUsage(input_tokens=5463, output_tokens=355, model='')
```


You can compute the cost for each run by passing in the cost per token to `Result.total_cost()` function.

In this case GPT-4o costs $5 for 1M input tokens and $15 for 1M output tokens.

```python
results.total_cost(cost_per_input_token=5 / 1e6, cost_per_output_token=15 / 1e6)
```

```py
0.03264
```
