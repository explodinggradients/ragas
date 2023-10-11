# Prepare data for evaluation

This tutorial notebook provides a step-by-step guide on how to prepare data for experimenting and evaluating using ragas. 

```{note}
If you're using popular frameworks like llama-index, langchain, etc to build your RAG application, Ragas provides integrations with these frameworks. Checkout [integrations](../integrations/)
```

This tutorial assumes that you have the 4 required data points from your RAG pipeline
1. Question: A set of questions. 
2. Contexts: Retrieved contexts corresponding to each question. This is a `list[list]` since each question can retrieve multiple text chunks.
3. Answer: Generated answer corresponding to each question.
4. Ground truths: Ground truths corresponding to each question. This is also a `list[list]` since each question may have multiple ground truths. 


## Example dataset

```{code-block} python
:caption: convert data samples to HF dataset
from datasets import Dataset 

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on January 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The Super Bowl....season since 1966,','replacing the NFL...in February.'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
    'ground_truths': [['The first superbowl was held on January 15, 1967'], ['The New England Patriots have won the Super Bowl a record six times']]
}
dataset = Dataset.from_dict(data_samples)
```