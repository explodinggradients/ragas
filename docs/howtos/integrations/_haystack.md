# Haystack
## Evaluating Haystack with Ragas

[Haystack](https://haystack.deepset.ai/) is an open-source framework for building production-ready LLM applications. The `RagasEvaluator` component evaluates Haystack Pipelines using LLM-based metrics. It supports metrics like context relevance, factual accuracy, response relevance, and more.


This guide will show you how to use the `RagasEvaluator` with a Haystack pipeline.

## Installation


```python
!pip install ragas-haystack
```

### Setting your `OPENAI_API_KEY` environment variable


Many metrics use OpenAI models and require an environment variable `OPENAI_API_KEY` to be set. If you don't have an OpenAI API key, you can sign up for an account here.


```python
import os
from getpass import getpass


os.environ["OPENAI_API_KEY"] = getpass("enter your OpenAI API key:")
```

    enter your OpenAI API key:··········


## Use the RagasEvaluator in a Haystack pipeline

To use the `RagasEvaluator` you need to follow these steps:

1. Initialize the `RagasEvaluator` while providing the correct `metric_params` for the metric you are using.
2. Run the `RagasEvaluator`, either on its own or in a Pipeline, by providing the expected input for the metric you are using.

To create a Pipeline that evaluates context relevance and aspect critique:


```python
from haystack import Pipeline
from haystack_integrations.components.evaluators.ragas import (
    RagasEvaluator,
    RagasMetric,
)

evaluation_pipeline = Pipeline()
evaluator_context = RagasEvaluator(
    metric=RagasMetric.CONTEXT_PRECISION,
)
evaluator_aspect = RagasEvaluator(
    metric=RagasMetric.ASPECT_CRITIQUE,
    metric_params={
        "name": "custom",
        "definition": "Is this answer problematic for children?",
        "strictness": 3,
    },
)
evaluation_pipeline.add_component("evaluator_context", evaluator_context)
evaluation_pipeline.add_component("evaluator_aspect", evaluator_aspect)
```

To run the evaluation Pipeline, you should have the expected inputs for the metrics ready at hand. These metrics expect a list of `questions`, `contexts`, `responses`, and `ground_truths`. These should come from the results of the `Pipeline` you want to evaluate.


```python
QUESTIONS = [
    "Which is the most popular global sport?",
    "Who created the Python language?",
]
CONTEXTS = [
    [
        "The popularity of sports can be measured in various ways, including TV viewership, social media presence, number of participants, and economic impact. Football is undoubtedly the world's most popular sport with major events like the FIFA World Cup and sports personalities like Ronaldo and Messi, drawing a followership of more than 4 billion people."
    ],
    [
        "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language. Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write clear, logical code for both small and large-scale software projects."
    ],
]
RESPONSES = [
    "Football is the most popular sport with around 4 billion followers worldwide",
    "Python language was created by Guido van Rossum.",
]
GROUND_TRUTHS = [
    "Football is the most popular sport",
    "Python language was created by Guido van Rossum.",
]

results = evaluation_pipeline.run(
    {
        "evaluator_context": {
            "questions": QUESTIONS,
            "contexts": CONTEXTS,
            "ground_truths": GROUND_TRUTHS,
        },
        "evaluator_aspect": {
            "questions": QUESTIONS,
            "contexts": CONTEXTS,
            "responses": RESPONSES,
        },
    }
)
```


```python
QUESTIONS = [
    "Which is the most popular global sport?",
    "Who created the Python language?",
]
CONTEXTS = [
    [
        "The popularity of sports can be measured in various ways, including TV viewership, social media presence, number of participants, and economic impact. Football is undoubtedly the world's most popular sport with major events like the FIFA World Cup and sports personalities like Ronaldo and Messi, drawing a followership of more than 4 billion people."
    ],
    [
        "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language. Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write clear, logical code for both small and large-scale software projects."
    ],
]
RESPONSES = [
    "Football is the most popular sport with around 4 billion followers worldwide",
    "Python language was created by Guido van Rossum.",
]
GROUND_TRUTHS = [
    "Football is the most popular sport",
    "Python language was created by Guido van Rossum.",
]
results = evaluation_pipeline.run(
    {
        "evaluator_context": {
            "questions": QUESTIONS,
            "contexts": CONTEXTS,
            "ground_truths": GROUND_TRUTHS,
        },
        "evaluator_aspect": {
            "questions": QUESTIONS,
            "contexts": CONTEXTS,
            "responses": RESPONSES,
        },
    }
)
```


```python
for component in ["evaluator_context", "evaluator_aspect"]:
    for output in results[component]["results"]:
        print(output)
```

    [{'name': 'context_precision', 'score': 0.9999999999}]
    [{'name': 'context_precision', 'score': 0.9999999999}]
    [{'name': 'custom', 'score': 0}]
    [{'name': 'custom', 'score': 0}]


You can use a Pandas dataframe to do additional analysis.


```python
import pandas as pd

df = pd.DataFrame.from_dict(results)
print(df)
```

                                             evaluator_context  \
    results  [[{'name': 'context_precision', 'score': 0.999...   
    
                                              evaluator_aspect  
    results  [[{'name': 'custom', 'score': 0}], [{'name': '...  

