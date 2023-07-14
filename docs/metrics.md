# Metrics

1. `faithfulness` : measures the factual consistency of the generated answer against the given context. This is done using a multi step paradigm that includes creation of statements from the generated answer followed by verifying each of these statements against the context. The answer is scaled to (0,1) range. Higher the better.
```python
from ragas.metrics.factuality import Faithfulness
faithfulness = Faithfulness()

# Dataset({
#     features: ['question','contexts','answer'],
#     num_rows: 25
# })
dataset: Dataset

results = faithfulness.score(dataset)
```

2. `context_relevancy`: measures how relevant is the retrieved context to the prompt. This is done using a combination of OpenAI models and cross-encoder models. To improve the score one can try to optimize the amount of information present in the retrieved context. 
```python
from ragas.metrics.context_relevancy import ContextRelevancy
context_rel = ContextRelevancy(strictness=3)
# Dataset({
#     features: ['question','contexts'],
#     num_rows: 25
# })
dataset: Dataset

results = context_rel.score(dataset)
```

3. `answer_relevancy`: measures how relevant is the generated answer to the prompt. This is quantified using conditional likelihood of an LLM generating the question given the answer. This is implemented using a custom model. Values range (0,1), higher the better.
```python
from ragas.metrics.answer_relevancy import AnswerRelevancy
answer_relevancy = AnswerRelevancy(model_name="t5-small")
# Dataset({
#     features: ['question','answer'],
#     num_rows: 25
# })
dataset: Dataset

results = answer_relevancy.score(dataset)
```


## Why is ragas better than scoring using GPT 3.5 directly.
LLM like GPT 3.5 struggle when it comes to scoring generated text directly. For instance, these models would always only generate integer scores and these scores vary when invoked differently. Advanced paradigms and techniques leveraging LLMs to minimize this bias is the solution ragas presents.
<h1 align="center">
  <img style="vertical-align:middle" height="350"
  src="./assets/bar-graph.svg">
</h1>

Take a look at our experiments [here](/experiments/assesments/metrics_assesments.ipynb)