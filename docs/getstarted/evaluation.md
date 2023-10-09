---
file_format: mystnb
kernelspec:
  name: python3
execution:
  timeout: 300
---
# Evaluation

welcome to the ragas quickstart. We're going to get you up and running with ragas as qickly as you can so that you can go back to improving your Retrieval Augmented Generation pipelines while this library makes sure your changes are improving your entire pipeline.

to kick things of lets start with the data

```{note}
Are you using Azure OpenAI endpoints? Then checkout [this quickstart guide](./guides/quickstart-azure-openai.ipynb)
```

```bash
pip install ragas
```

Ragas also uses OpenAI for running some metrics so make sure you have your openai key ready and available in your environment
```python
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"
```
## The Data

Ragas performs a `ground_truth` free evaluation of your RAG pipelines. This is because for most people building a gold labeled dataset which represents in the distribution they get in production is a very expensive process.

```{note}
While originially ragas was aimed at `ground_truth` free evalutions there is some aspects of the RAG pipeline that need `ground_truth` in order to measure. We're in the process of building a testset generation features that will make it easier. Checkout [issue#136](https://github.com/explodinggradients/ragas/issues/136) for more details.
```

Hence to work with ragas all you need are the following data
- question: `list[str]` - These are the questions you RAG pipeline will be evaluated on. 
- answer: `list[str]` - The answer generated from the RAG pipeline and give to the user.
- contexts: `list[list[str]]` - The contexts which where passed into the LLM to answer the question.
- ground_truths: `list[list[str]]` - The ground truth answer to the questions. (only required if you are using context_recall)

Ideally your list of questions should reflect the questions your users give, including those that you have been problamatic in the past.

Here we're using an example dataset from on of the baselines we created for the [Financial Opinion Mining and Question Answering (fiqa) Dataset](https://sites.google.com/view/fiqa/) we created. If you want to want to know more about the baseline, feel free to check the `experiements/baseline` section

```{code-cell} python
# data
from datasets import load_dataset

fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")
fiqa_eval
```

## Metrics

Ragas provides you with a few metrics to evaluate the different aspects of your RAG systems namely

1. metrics to evaluate retrieval: offers `context_precision` and `context_recall` which give you the measure of the performance of your retrieval system. 
2. metrics to evaluate generation: offers `faithfulness` which measures hallucinations and `answer_relevancy` which measures how to the point the answers are to the question.

The harmonic mean of these 4 aspects gives you the **ragas score** which is a single measure of the performance of your QA system across all the important aspects.

now lets import these metrics and understand more about what they denote

```{code-cell}
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness
```
here you can see that we are using 4 metrics, but what do the represent?

1. context_precision - a measure of how relevant the retrieved context is to the question. Conveys quality of the retrieval pipeline.
2. answer_relevancy - a measure of how relevent the answer is to the question
3. faithfulness - the factual consistancy of the answer to the context base on the question.
4. context_recall: measures the ability of the retriever to retrieve all the necessary information needed to answer the question. 
5. harmfulness (AspectCritique) - in general, `AspectCritique` is a metric that can be used to quantify various aspects of the answer. Aspects like harmfulness, maliciousness, coherence, correctness, concisenes are available by default but you can easily define your own. Check the [docs](./metrics.md) for more info.

```{note}
by default these metrics are using OpenAI's API to compute the score. If you using this metric make sure you set the environment key `OPENAI_API_KEY` with your API key. You can also try other LLMs for evaluation, check the [llm guide](./guides/llms.ipynb) to learn more
```

If you're interested in learning more, feel free to check the [docs](https://github.com/explodinggradients/ragas/blob/main/docs/metrics.md)

## Evaluation

Running the evalutation is as simple as calling evaluate on the `Dataset` with the metrics of your choice.

```{code-cell}
from ragas import evaluate

result = evaluate(
    fiqa_eval["baseline"].select(range(1)),
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        harmfulness,
    ],
)

result
```
and there you have the it, all the scores you need. `ragas_score` gives you a single metric that you can use while the other onces measure the different parts of your pipeline.

now if we want to dig into the results and figure out examples where your pipeline performed worse or really good you can easily convert it into a pandas array and use your standard analytics tools too!

```{code-cell}
df = result.to_pandas()
df.head()
```
And thats it!

if you have any suggestion/feedbacks/things your not happy about, please do share it in the [issue section](https://github.com/explodinggradients/ragas/issues). We love hearing from you 😁
