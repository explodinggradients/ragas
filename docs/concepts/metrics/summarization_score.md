# Summarization Score

This metric gives a measure of how well the `summary` captures the important information from the `contexts`. The intuition behind this metric is that a good summary shall contain all the important information present in the context(or text so to say).

We first extract a set of important keyphrases from the context. These keyphrases are then used to generate a set of questions. The answers to these questions are always `yes(1)` for the context. We then ask these questions to the summary and calculate the summarization score as the ratio of correctly answered questions to the total number of questions. 

We compute the question-answer score using the answers, which is a list of `1`s and `0`s. The question-answer score is then calculated as the ratio of correctly answered questions(answer = `1`) to the total number of questions.

```{math}
:label: question-answer-score
\text{QA score} = \frac{|\text{correctly answered questions}|}{|\text{total questions}|}
````

We also introduce an option to penalize larger summaries by proving a conciseness score. If this option is enabled, the final score is calculated as the average of the summarization score and the conciseness score. This conciseness scores ensures that summaries that are just copies of the text do not get a high score, because they will obviously answer all questions correctly.

```{math}
:label: conciseness-score
\text{conciseness score} = \frac{\text{length of summary}}{\text{length of context}}
````

The final summarization score is then calculated as:

```{math}
:label: summarization-score
\text{Summarization Score} = \frac{\text{QA score} + \text{conciseness score}}{2}
````

```{hint}
**Summary**: JPMorgan Chase & Co. is an American multinational finance company headquartered in New York City. It is the largest bank in the United States and the world's largest by market capitalization as of 2023. Founded in 1799, it is a major provider of investment banking services, with US$3.9 trillion in total assets, and ranked #1 in the Forbes Global 2000 ranking in 2023.


**keyphrases**: [
                "JPMorgan Chase & Co.",\
                "American multinational finance company",\
                "headquartered in New York City",\
                "largest bank in the United States",\
                "world's largest bank by market capitalization",\
                "founded in 1799",\
                "major provider of investment banking services",\
                "US$3.9 trillion in total assets",\
                "ranked #1 in Forbes Global 2000 ranking",\
            ]

**Questions**: [
                "Is JPMorgan Chase & Co. an American multinational finance company?",\
                "Is JPMorgan Chase & Co. headquartered in New York City?",\
                "Is JPMorgan Chase & Co. the largest bank in the United States?",\
                "Is JPMorgan Chase & Co. the world's largest bank by market capitalization as of 2023?",\
                "Is JPMorgan Chase & Co. considered systemically important by the Financial Stability Board?",\
                "Was JPMorgan Chase & Co. founded in 1799 as the Chase Manhattan Company?",\
                "Is JPMorgan Chase & Co. a major provider of investment banking services?",\
                "Is JPMorgan Chase & Co. the fifth-largest bank in the world by assets?",\
                "Does JPMorgan Chase & Co. operate the largest investment bank by revenue?",\
                "Was JPMorgan Chase & Co. ranked #1 in the Forbes Global 2000 ranking?",\
                "Does JPMorgan Chase & Co. provide investment banking services?",\
            ]

**Answers**: ["0", "1", "1", "1", "0", "0", "1", "1", "1", "1", "1"]
````

## Example

```{code-block} python
from datasets import Dataset 
from ragas.metrics import summarization_score
from ragas import evaluate

data_samples = {
    'contexts' : [[c1], [c2]],
    'summary': [s1, s2]
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[summarization_score])
score.to_pandas()
```

