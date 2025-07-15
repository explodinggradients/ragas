# Prompt Evaluation

In this tutorial, we will write a simple evaluation pipeline to evaluate a prompt that is part of an AI system, here a movie review sentiment classifier. At the end of this tutorial you’ll learn how to evaluate and iterate on a single prompt using evaluation driven development. 

```mermaid
flowchart LR
    A["'This movie was amazing!<br/>Great acting and plot.'"] --> B["Classifier Prompt"]
    B --> C["Positive"]
```


We will start by writing a simple prompt that classifies movie reviews as positive or negative. 

```python
SYSTEM_PROMPT = """
You are a movie review classifier. You classify movie reviews as positive or negative.
"""
```

Next, we will write down few sample inputs and expected outputs for our prompt. Then convert them to a a csv file

```python
import pandas as pd

samples = [{"text": "I loved the movie! It was fantastic.", "label": "positive"},
    {"text": "The movie was terrible and boring.", "label": "negative"},
    {"text": "It was an average film, nothing special.", "label": "positive"},
    {"text": "Absolutely amazing! Best movie of the year.", "label": "positive"}]
pd.DataFrame(samples).to_csv("datasets/test_dataset.csv", index=False)
```

Now we need to have a way to measure the performance of our prompt in this task. We will define a metric that will compare the output of our prompt with the expected output and outputs pass/fail based on it. 

```python
from ragas_experimental.metric import discrete_metric
from ragas_experimental.metric.result import MetricResult

@discrete_metric(name="accuracy", values=["pass", "fail"])
def my_metric(prediction: str, actual: str):
    """Calculate accuracy of the prediction."""
    return MetricResult(result="pass", reason="") if prediction == actual else MetricResult(result="fail", reason="")
```

Next, we will write the evaluation function that will run our prompt on the test dataset and evaluate it using the metric, and store the results in a csv file. 

```python
from ragas_experimental import experiment

@experiment()
async def run_experiment(row):
    
    response = run_prompt(row["text"])
    score = my_metric.score(
        prediction=response,
        actual=row["label"]
    )

    experiment_view = {
        **row,
        "response":response,
        "score":score.result,
    }
    return experiment_view
```



## Running the example end to end

1. Setup your OpenAI API key

```bash
export OPENAI_API_KEY = "your_openai_api_key"
```

2. Run the evaluation

```bash
python -m ragas_examples.prompt_evals.evals
```

Voila! You have successfully run your first evaluation using Ragas. You can now inspect the results by opening the `experiments/experiment_name.csv` file. 

