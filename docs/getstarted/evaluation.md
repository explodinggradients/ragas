(get-started-evaluation)=
# Evaluate Your Testset

Welcome to the Ragas quickstart. Our aim is to get you up and running with Ragas as quickly as possible, so that you can focus on improving your Retrieval Augmented Generation pipelines while this library ensures your changes are enhancing your entire pipeline.

To kick things off, let's start with the data.

:::{note}
Are you using Azure OpenAI endpoints? Then check out [this quickstart guide](../howtos/customisations/azure-openai.ipynb).
:::

```bash
pip install ragas
```

Ragas also uses OpenAI for running some metrics, so ensure you have your OpenAI key ready and available in your environment.

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
```
## The Data

For this tutorial, we are going to use an example dataset from one of the baselines we created for the [Financial Opinion Mining and Question Answering (FIQA) Dataset](https://sites.google.com/view/fiqa/). The dataset has the following columns:

- question: `list[str]` - These are the questions your RAG pipeline will be evaluated on.
- answer: `list[str]` - The answer generated from the RAG pipeline and given to the user.
- contexts: `list[list[str]]` - The contexts which were passed into the LLM to answer the question.
- ground_truths: `list[list[str]]` - The ground truth answer to the questions. (only required if you are using context_recall)

Ideally, your list of questions should reflect the questions your users ask, including those that have been problematic in the past.

```{code-block} python
:caption: import sample dataset
from datasets import load_dataset

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
amnesty_qa
```

:::{seealso}
See [testset generation](./testset_generation.md) to learn how to generate your own synthetic data for evaluation.
:::

## Metrics

Ragas provides you with a few metrics to evaluate the different aspects of your RAG systems. 

1. Retriever: Offers `context_precision` and `context_recall` which measure the performance of your retrieval system.
2. Generator (LLM): Offers `faithfulness` which measures hallucinations and `answer_relevancy` which measures how to the point the answers are to the question.

The harmonic mean of these 4 aspects gives you the **Ragas score** which is a single measure of the performance of your QA system across all the important aspects.

Now, let's import these metrics and understand more about what they denote.

```{code-block} python
:caption: import metrics
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
```
Here you can see that we are using 4 metrics, but what do they represent?

1. Faithfulness - The factual consistency of the answer to the context based on the question.
2. Context_precision - A measure of how relevant the retrieved context is to the question. Conveys quality of the retrieval pipeline.
3. Answer_relevancy - A measure of how relevant the answer is to the question.
4. Context_recall - Measures the ability of the retriever to retrieve all the necessary information needed to answer the question.

:::{note}
By default, these metrics are using OpenAI's API to compute the score. If you are using this metric, make sure you set the environment key `OPENAI_API_KEY` with your API key. You can also try other LLMs for evaluation, check the [LLM guide](../howtos/customisations/llms.ipynb) to learn more.
:::

## Evaluation

Running the evaluation is as simple as calling `evaluate` on the `Dataset` with the metrics of your choice.

```{code-block} python
:caption: evaluate using sample dataset
from ragas import evaluate

result = evaluate(
    amnesty_qa["eval"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

result
```
And there you have it, all the scores you need.

If you want to dig into the results and figure out examples where your pipeline performed poorly or exceptionally well, you can easily convert it into a pandas DataFrame and use your standard analytics tools too!

```{code-block} python
:caption: export results
df = result.to_pandas()
df.head()
```
<p align="left">
<img src="../_static/imgs/quickstart-output.png" alt="quickstart-outputs" width="800" height="600" />
</p>

And that's it!

If you have any suggestions, feedback, or issues, please share them in the [issue section](https://github.com/explodinggradients/ragas/issues). We value your input.
