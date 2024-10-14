# Understand Cost and Usage of Operations

When using LLMs for evaluation and test set generation, cost will be an important factor. Ragas provides you some tools to help you with that.

## Understanding `TokenUsageParser`

By default Ragas does not calculate the usage of tokens for `evaluate()`. This is because langchain's LLMs do not always return information about token usage in a uniform way. So in order to get the usage data, we have to implement a `TokenUsageParser`. 

A `TokenUsageParser` is function that parses the `LLMResult` or `ChatResult` from langchain models `generate_prompt()` function and outputs `TokenUsage` which Ragas expects.

For an example here is one that will parse OpenAI by using a parser we have defined.


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




    TokenUsage(input_tokens=9, output_tokens=9, model='')



You can define your own or import parsers if they are defined. If you would like to suggest parser for LLM providers or contribute your own ones please check out this [issue](https://github.com/explodinggradients/ragas/issues/1151) 🙂.

You can use it for evaluations as so. Using example from [get started](get-started-evaluation) here.


```python
from datasets import load_dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
amnesty_qa
```

    Repo card metadata block was not found. Setting CardData to empty.





    DatasetDict({
        eval: Dataset({
            features: ['question', 'ground_truth', 'answer', 'contexts'],
            num_rows: 20
        })
    })




```python
from ragas import evaluate
from ragas.cost import get_token_usage_for_openai

result = evaluate(
    amnesty_qa["eval"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    llm=gpt4o,
    token_usage_parser=get_token_usage_for_openai,
)
```


    Evaluating:   0%|          | 0/80 [00:00<?, ?it/s]



```python
result.total_tokens()
```




    TokenUsage(input_tokens=116765, output_tokens=39031, model='')



You can compute the cost for each run by passing in the cost per token to `Result.total_cost()` function.

In this case GPT-4o costs $5 for 1M input tokens and $15 for 1M output tokens.


```python
result.total_cost(cost_per_input_token=5 / 1e6, cost_per_output_token=15 / 1e6)
```




    1.1692900000000002


