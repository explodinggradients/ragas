# Domain Specific Evaluation

Domain specific evaluation metric is a rubric-based evaluation metric that is used to evaluate the performance of a model on a specific domain. The rubric consists of descriptions for each score, typically ranging from 1 to 5. The response here is evaluation and scored using the LLM using description specified in the rubric. This metric also have reference free and reference based variations. 

For example, in RAG if you have the `question`, `contexts`, `answer` and `ground_truth` (optional) then you can decide the rubric based on the domain (or use the default rubrics provided by ragas) and evaluate the model using this metric. 

## Example


```python
from ragas import evaluate
from datasets import Dataset, DatasetDict

from ragas.metrics import rubrics_score_without_reference, rubrics_score_with_reference

rows = {
    "question": [
        "What's the longest river in the world?",
    ],
    "ground_truth": [
        "The Nile is a major north-flowing river in northeastern Africa.",
    ],
    "answer": [
        "The longest river in the world is the Nile, stretching approximately 6,650 kilometers (4,130 miles) through northeastern Africa, flowing through countries such as Uganda, Sudan, and Egypt before emptying into the Mediterranean Sea. There is some debate about this title, as recent studies suggest the Amazon River could be longer if its longest tributaries are included, potentially extending its length to about 7,000 kilometers (4,350 miles).",
    ],
    "contexts": [
        [
            "Scientists debate whether the Amazon or the Nile is the longest river in the world. Traditionally, the Nile is considered longer, but recent information suggests that the Amazon may be longer.",
            "The Nile River was central to the Ancient Egyptians' rise to wealth and power. Since rainfall is almost non-existent in Egypt, the Nile River and its yearly floodwaters offered the people a fertile oasis for rich agriculture.",
            "The world's longest rivers are defined as the longest natural streams whose water flows within a channel, or streambed, with defined banks.",
            "The Amazon River could be considered longer if its longest tributaries are included, potentially extending its length to about 7,000 kilometers."
        ],
    ]
}



dataset = Dataset.from_dict(rows)

result = evaluate(
    dataset,
    metrics=[
        rubrics_score_without_reference,
        rubrics_score_with_reference
    ],
)

```

Here the evaluation is done using both reference free and reference based rubrics. You can also declare and use your own rubric by defining the rubric in the `rubric` parameter.

```python
from ragas.metrics.rubrics import RubricsScoreWithReference

my_custom_rubrics = {
    "score1_description": "answer and ground truth are completely different",
    "score2_description": "answer and ground truth are somewhat different",
    "score3_description": "answer and ground truth are somewhat similar",
    "score4_description": "answer and ground truth are similar",
    "score5_description": "answer and ground truth are exactly the same",
}

rubrics_score_with_reference = RubricsScoreWithReference(rubrics=my_custom_rubrics)
```


