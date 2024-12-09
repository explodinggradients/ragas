# Traditional NLP Metrics

## Non LLM String Similarity

`NonLLMStringSimilarity` metric measures the similarity between the reference and the response using traditional string distance measures such as Levenshtein, Hamming, and Jaro. This metric is useful for evaluating the similarity of `response` to the `reference` text without relying on large language models (LLMs). The metric returns a score between 0 and 1, where 1 indicates a perfect match between the response and the reference. This is a non LLM based metric.

### Example
```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._string import NonLLMStringSimilarity

sample = SingleTurnSample(
    response="The Eiffel Tower is located in India.",
    reference="The Eiffel Tower is located in Paris."
)

scorer = NonLLMStringSimilarity()
await scorer.single_turn_ascore(sample)
```

One can choose from available string distance measures from `DistanceMeasure`. Here is an example of using Hamming distance.

```python
from ragas.metrics._string import NonLLMStringSimilarity, DistanceMeasure

scorer = NonLLMStringSimilarity(distance_measure=DistanceMeasure.HAMMING)
```


## BLEU Score

The `BleuScore` score is a metric used to evaluate the quality of `response` by comparing it with `reference`. It measures the similarity between the response and the reference based on n-gram precision and brevity penalty. BLEU score was originally designed to evaluate machine translation systems, but it is also used in other natural language processing tasks. BLEU score ranges from 0 to 1, where 1 indicates a perfect match between the response and the reference. This is a non LLM based metric.

### Example
```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore

sample = SingleTurnSample(
    response="The Eiffel Tower is located in India.",
    reference="The Eiffel Tower is located in Paris."
)

scorer = BleuScore()
await scorer.single_turn_ascore(sample)
```


## ROUGE Score

The `RougeScore` score is a set of metrics used to evaluate the quality of natural language generations. It measures the overlap between the generated `response` and the `reference` text based on n-gram recall, precision, and F1 score. ROUGE score ranges from 0 to 1, where 1 indicates a perfect match between the response and the reference. This is a non LLM based metric.

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import RougeScore

sample = SingleTurnSample(
    response="The Eiffel Tower is located in India.",
    reference="The Eiffel Tower is located in Paris."
)

scorer = RougeScore()
await scorer.single_turn_ascore(sample)
```

You can change the `rouge_type` to `rouge-1`, `rouge-2`, or `rouge-l` to calculate the ROUGE score based on unigrams, bigrams, or longest common subsequence respectively.

```python
scorer = RougeScore(rouge_type="rouge-1")
```

You can change the `measure_type` to `precision`, `recall`, or `f1` to calculate the ROUGE score based on precision, recall, or F1 score respectively.

```python
scorer = RougeScore(measure_type="recall")
```

## Exact Match
The `ExactMatch` metric checks if the response is exactly the same as the reference text. It is useful in scenarios where you need to ensure that the generated response matches the expected output word-for-word. For example, arguments in tool calls, etc. The metric returns 1 if the response is an exact match with the reference, and 0 otherwise.

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ExactMatch

sample = SingleTurnSample(
    response="India",
    reference="Paris"
)

scorer = ExactMatch()
await scorer.single_turn_ascore(sample)
```

## String Presence
The `StringPresence` metric checks if the response contains the reference text. It is useful in scenarios where you need to ensure that the generated response contains certain keywords or phrases. The metric returns 1 if the response contains the reference, and 0 otherwise.

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import StringPresence

sample = SingleTurnSample(
    response="The Eiffel Tower is located in India.",
    reference="Eiffel Tower"
)
scorer = StringPresence()
await scorer.single_turn_ascore(sample)
```
