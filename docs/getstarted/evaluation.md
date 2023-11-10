(get-started-evaluation)=
# Evaluation

Welcome to the ragas quickstart. We're going to get you up and running with ragas as quickly as you can so that you can go back to improving your Retrieval Augmented Generation pipelines while this library makes sure your changes are improving your entire pipeline.

to kick things of lets start with the data

:::{note}
Are you using Azure OpenAI endpoints? Then checkout [this quickstart
guide](../howtos/customisations/azure-openai.ipynb)
:::

```bash
pip install ragas
```

Ragas also uses OpenAI for running some metrics so make sure you have your openai key ready and available in your environment

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
```
## The Data

For this tutorial we are going to use an example dataset from one of the baselines we created for the [Financial Opinion Mining and Question Answering (fiqa) Dataset](https://sites.google.com/view/fiqa/). The dataset has the following columns.

- question: `list[str]` - These are the questions your RAG pipeline will be evaluated on.
- answer: `list[str]` - The answer generated from the RAG pipeline and given to the user.
- contexts: `list[list[str]]` - The contexts which were passed into the LLM to answer the question.
- ground_truths: `list[list[str]]` - The ground truth answer to the questions. (only required if you are using context_recall)

Ideally your list of questions should reflect the questions your users give, including those that you have been problematic in the past.


```{code-block} python
:caption: import sample dataset
from datasets import load_dataset

fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")
fiqa_eval
```

:::{seealso}
See [testset generation](./testset_generation.md) to learn how to generate your own synthetic data for evaluation.
:::

## Metrics

Ragas provides you with a few metrics to evaluate the different aspects of your RAG systems namely

1. Retriever: offers `context_precision` and `context_recall` which give you the measure of the performance of your retrieval system.
2. Generator (LLM): offers `faithfulness` which measures hallucinations and `answer_relevancy` which measures how to the point the answers are to the question.

The harmonic mean of these 4 aspects gives you the **ragas score** which is a single measure of the performance of your QA system across all the important aspects.

now lets import these metrics and understand more about what they denote

```{code-block} python
:caption: import metrics
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
```
here you can see that we are using 4 metrics, but what do the represent?

1. faithfulness - the factual consistency of the answer to the context base on the question.
2. context_precision - a measure of how relevant the retrieved context is to the question. Conveys quality of the retrieval pipeline.
3. answer_relevancy - a measure of how relevant the answer is to the question
4. context_recall: measures the ability of the retriever to retrieve all the necessary information needed to answer the question.


:::{note}
by default these metrics are using OpenAI's API to compute the score. If you using this metric make sure you set the environment key `OPENAI_API_KEY` with your API key. You can also try other LLMs for evaluation, check the [llm guide](../howtos/customisations/llms.ipynb) to learn more
:::

## Evaluation

Running the evaluation is as simple as calling evaluate on the `Dataset` with the metrics of your choice.

```{code-block} python
:caption: evaluate using sample dataset
from ragas import evaluate

result = evaluate(
    fiqa_eval["baseline"].select(range(3)), # selecting only 3
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

result
```
and there you have the it, all the scores you need. `ragas_score` gives you a single metric that you can use while the other onces measure the different parts of your pipeline.

now if we want to dig into the results and figure out examples where your pipeline performed worse or really good you can easily convert it into a pandas array and use your standard analytics tools too!

```{code-block} python
:caption: export results
df = result.to_pandas()
df.head()
```
<p align="left">
<img src="../_static/imgs/quickstart-output.png" alt="quickstart-outputs" width="800" height="600" />
</p>

And thats it!

if you have any suggestion/feedbacks/things your not happy about, please do share it in the [issue section](https://github.com/explodinggradients/ragas/issues). We love hearing from you 😁
