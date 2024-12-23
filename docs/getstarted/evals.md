# Get Started

The purpose of this guide is to illustrate a simple workflow for testing and evaluating an LLM application with `ragas`. It assumed minimum knowledge in AI application building and evaluation. Please refer to our [installation instruction](./install.md) for installing `ragas`


## Evaluation


!!! note
    For the sake of this guide, you will be evaluating a text summarization pipeline. The goal is to ensure that the output summary captures all the important details specified in text such as growth numbers, market details, etc.

`ragas` provides a dozen of methods for analyzing the performance of an LLM application, called [metrics](../concepts/metrics/). Each metric expects a predefined set of data points using which it calculates scores indicating performance. 

### Evaluating using a Non-LLM Metric

Here is a simple example that uses `BleuScore` score to score summary

```python
from ragas import SingleTurnSample
from ragas.metrics import BleuScore

test_data = {
    "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
    "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    "reference": "The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter."
}
metric = BleuScore()
test_data = SingleTurnSample(**test_data)
metric.single_turn_score(test_data)
```

```
0.137
```

Here we used,

- A test sample containing `user_input`, `response` (the output from LLM), and `reference` (the expected output from LLM) as data points to evaluate the summary.
- A non llm metric called `BleuScore`

As you may observe, this has two shortcomings

- Evaluating the application requires you to prepare the expected output against each input, which can be time consuming and hard.
- Even though the `response` and `reference` are similar, the output score was low. This is a known limitation of non llm metrics like Blue Score. Here by non-LLM metric, we mean a metric that does not use LLM to evaluate.

To tackle this, let's try an LLM based metric without reference next.


### Evaluating using a LLM based Metric


**Choose your LLM**
--8<--
choose_evaluator_llm.md
--8<--

**Evaluation**


Here we will use `AspectCritic`, which an LLM based metric that outputs pass/fail given the evaluation criteria.


```python
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic

test_data = {
    "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
    "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
}

metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Verify if the summary is accurate.")
test_data = SingleTurnSample(**test_data)
await metric.single_turn_ascore(test_data)

```

```
1
```

Success! Here 1 means pass and 0 means fail

### Evaluating on a Dataset

In both of the above examples, we have only used one sample to evaluate our application, which is not very robust. To make sure the evaluation is robust enough to trust, we can add few more test samples to our test data. Here I am loading dataset from HF, one may load it from any source like production logs, etc. Just make sure that each sample has all the required attributes for the chosen metric. Here in our case it's `user_input` and `reference`. 

```python
from datasets import load_dataset
eval_dataset = load_dataset("explodinggradients/earning_report_summary")
eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)
eval_dataset
```

```
EvaluationDataset(features=['user_input', 'response'], len=50)
```

Evaluate using dataset

```python
results = evaluate(test_data, metrics=[metric])
results
```

```
{'summary_accuracy': 0.84}
```

This score shows that out of all the samples in our test data, only 84% of summaries passes the given evaluation criteria.

Export the sample level scores to pandas dataframe

```python
results.to_pandas()
```


Viewing the sample level results in a csv as shown above is okayish, but not great. It makes it hard to view and analyze each sample, and compare results across evaluation runs, etc. For that you may use [app.ragas.io](https://app.ragas.io/)


## Analyzing Results

For this you may sign up and setup [app.ragas.io]() easily. If not, you may use any alternative tools available to you. 

In order to use the [app.ragas.io](http://app.ragas.io) dashboard, you need to have an account on [app.ragas.io](https://app.ragas.io/). If you don't have one, you can sign up for one [here](https://app.ragas.io/login). You will also need to generate a [Ragas API key](https://app.ragas.io/settings/api-keys).

Once you have the API key, you can use the `upload()` method to export the results to the dashboard.

```python
import os
os.environ["RAGAS_API_KEY"] = "your_api_key"
```

Now you can view the results in the dashboard by following the link in the output of the `upload()` method.

```python
results.upload()
```

![](../_static/imgs/ragas_get_started_evals.gif)


## Aligning metrics

When you're viewing the evaluation results in [app.ragas.io](https://app.ragas.io/), you may notice that sometimes the LLM based metric makes mistakes while evaluating the application using given criteria. For example, 

![](../_static/imgs/eval_mistake1.png)

Even though the `response` summary drops most of the important information such as growth numbers, market domain, etc the LLM based metric mistakenly marks it as accurate which is not what we want. You may observe many other samples like this. 

To fix these results, you need to align the LLM based metric with your preferences so that ragas `metric` learns your preferences like a machine learning model. To make this possible, ragas has a feature to train your own metric using preference data collected from the app. Here the two step process for this,

- Accept, Reject and Edit evaluation results to form the train data ( Atleast 15-20 samples)
- Download the annotated data using `Annotated JSON` button
- Train the metric

[Download sample annotated JSON](../_static/sample_annotated_summary.json)

```python
from ragas.config import InstructionConfig,DemonstrationConfig
demo_config = DemonstrationConfig(embedding=evaluator_embeddings)
inst_config = InstructionConfig(llm=evaluator_llm)

evaluator.train(path="<your-annotated-json.json>",demonstration_config=demo_config,instruction_config=inst_config)
```

Once trained, you may redo the evaluation on same or different test data sets. You will now observe that the metric has learned your preferences like a machine learning model and now makes less mistakes while evaluating.

## Up Next

- [Run ragas metrics for evaluating RAG](rag_evaluation.md)
- [Generate test data for evaluating RAG](rag_testset_generation.md)