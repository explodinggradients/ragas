# Answer Correctness

The assessment of Answer Correctness involves gauging the accuracy of the generated answer when compared to the ground truth. This evaluation relies on the `ground truth` and the `answer`, with scores ranging from 0 to 1. A higher score indicates a closer alignment between the generated answer and the ground truth, signifying better correctness.

Answer correctness encompasses two critical aspects: semantic similarity between the generated answer and the ground truth, as well as factual similarity. These aspects are combined using a weighted scheme to formulate the answer correctness score. Users also have the option to employ a 'threshold' value to round the resulting score to binary, if desired.


```{hint}
Ground truth: Einstein was born in 1879 at Germany .

High answer correctness: In 1879, in Germany, Einstein was born. 

Low answer correctness: In Spain, Einstein was born in 1879. 
```


## Example

```{code-block} python
:caption: Answer correctness with custom weights for each variable
from ragas.metrics import AnswerCorrectness
answer_correctness = AnswerCorrectness(
    weights=[0.4,0.6]
)

# Dataset({
#     features: ['answer','ground_truths'],
#     num_rows: 25
# })
dataset: Dataset

results = answer_correctness.score(dataset)

```