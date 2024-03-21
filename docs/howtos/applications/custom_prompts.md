## Write custom prompts with ragas

This is a tutorial notebook that shows how to create and use custom prompts with the metrics used in the evaluation task. This is achieved using Ragas prompt class. This tutorial will guide you to change and use different prompts with the Ragas metrics instead of the default ones used.

**Dataset**

Here I’m using a dataset from HuggingFace.

```{code-block} python

from datasets import load_dataset, Dataset

amnesty_dataset = load_dataset("explodinggradients/amnesty_qa", "english")
amnesty_dataset
```

```{code-block} bash
DatasetDict({
    train: Dataset({
        features: ['question', 'ground_truth', 'answer', 'contexts'],
        num_rows: 20
    })
})
```

**Create a Custom Prompt Object**

Create a new Prompt object to be used in the metric for the evaluation task. For this task, I will be instantiating an object of the Ragas Prompt class.

```{code-block} python
from ragas.llms.prompt import Prompt

long_form_answer_prompt_new = Prompt(
    name="long_form_answer_new_v1",
    instruction="Create one or more statements from each sentence in the given answer.",
    examples=[
        {
            "question": "Which is the only planet in the solar system that has life on it?",
            "answer": "earth",
            "statements": {
                "statements": [
                    "Earth is the only planet in the solar system that has life on it."
                ]
            },
        },
        {
            "question": "Were Hitler and Benito Mussolini of the same nationality?",
            "answer": "Sorry, I can't provide an answer to that question.",
            "statements": {
                "statements": []
            },
        },
    ],
    input_keys=["question", "answer"],
    output_key="statements",
    output_type="json",
)
```

**Using the Custom Prompt in Evaluations**

I will be using the **faithfulness** metric for my evaluation task. Faithfulness uses two default prompts `long_form_answer_prompt` and `nli_statements_message` for evaluations. I will be changing the default `long_form_answer_prompt` used in this metric to the newly created prompt object.

```{code-block} python
from ragas.metrics import faithfulness

faithfulness.long_form_answer_prompt = long_form_answer_prompt_new
print(faithfulness.long_form_answer_prompt.to_string())
```

```{code-block} bash
Create one or more statements from each sentence in the given answer.

question: "Which is the only planet in the solar system that has life on it?"
answer: "earth"
statements: {{"statements": ["Earth is the only planet in the solar system that has life on it."]}}

question: "Were Hitler and Benito Mussolini of the same nationality?"
answer: "Sorry, I can't provide an answer to that question."
statements: {{"statements": []}}

question: {question}
answer: {answer}
statements:
```

Now the custom prompt that we created is being used in the faithfulness metric. We can now evaluate the dataset against the metric that uses the new prompt that we created.

```{code-block} python
from ragas import evaluate

result = evaluate(
    dataset["train"].select(range(3)), # selecting only 3
    metrics=[
        faithfulness
    ],
)

result
```
```{code-block} bash
evaluating with [faithfulness]
100%|██████████| 1/1 [02:31<00:00, 151.79s/it]

{'faithfulness': 0.7879}
```
