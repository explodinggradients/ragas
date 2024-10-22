## Run ragas metrics for evaluating RAG

In this tutorial, we will take a sample [test dataset](https://huggingface.co/datasets/explodinggradients/amnesty_qa), select a few of the [available metrics](../concepts/metrics/available_metrics/index.md) that Ragas offers, and evaluate a simple RAG pipeline. 

### Working with Data

The dataset used here is from [Amnesty QA RAG](https://huggingface.co/datasets/explodinggradients/amnesty_qa) that contains the necessary data points we need for this tutorial. Here I am loading it from huggingface hub, but you may use file from any source. 

```python
from datasets import load_dataset
dataset = load_dataset("explodinggradients/amnesty_qa","english_v3")
```

Load the dataset into Ragas EvaluationDataset object. 

```python
from ragas import EvaluationDataset

eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])
```


### Selecting required metrics
Ragas offers a [wide variety of metrics](../concepts/metrics/available_metrics/index.md/#retrieval-augmented-generation) that one can select from to evaluate LLM applications. You can also build your own metrics on top of ragas. For this tutorial, we will select a few metrics that are commonly used to evaluate single turn RAG systems.

```python
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
```

Since all of the metrics we have chosen are LLM-based metrics, we need to choose the evaluator LLMs we want to use for evaluation.

### Choosing evaluator LLM

--8<--
choose_evaluator_llm.md
--8<--


### Running Evaluation

```python
metrics = [
    LLMContextRecall(llm=evaluator_llm), 
    FactualCorrectness(llm=evaluator_llm), 
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]
results = evaluate(dataset=eval_dataset, metrics=metrics)
```

### Exporting and analyzing results

```python
df = results.to_pandas()
df.head()
```

![evaluation-result](./raga_evaluation_output.png)