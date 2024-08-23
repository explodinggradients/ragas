# Summarization Score

This metric gives a measure of how well the `summary` captures the important information from the `contexts`. The intuition behind this metric is that a good summary shall contain all the important information present in the context(or text so to say).

We first extract a set of important keyphrases from the context. These keyphrases are then used to generate a set of questions. The answers to these questions are always `yes(1)` for the context. We then ask these questions to the summary and calculate the summarization score as the ratio of correctly answered questions to the total number of questions. 

We compute the question-answer score using the answers, which is a list of `1`s and `0`s. The question-answer score is then calculated as the ratio of correctly answered questions(answer = `1`) to the total number of questions.

```{math}
:label: question-answer-score
\text{QA score} = \frac{|\text{correctly answered questions}|}{|\text{total questions}|}
````

We also introduce an option to penalize larger summaries by proving a conciseness score. If this option is enabled, the final score is calculated as the weighted average of the summarization score and the conciseness score. This conciseness scores ensures that summaries that are just copies of the text do not get a high score, because they will obviously answer all questions correctly. Also, we do not want the summaries that are empty. We add a small value `1e-10` to the denominator to avoid division by zero.

```{math}
:label: conciseness-score
\text{conciseness score} = 1 - \frac{\min(\text{length of summary}, \text{length of context})}{\text{length of context} + \text{1e-10}}
````

We also provide a coefficient `coeff`(default value 0.5) to control the weightage of the scores. 

The final summarization score is then calculated as:

```{math}
:label: summarization-score
\text{Summarization Score} = \text{QA score}*\text{coeff} + \\
\text{conciseness score}*\text{(1-coeff)}
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
from ragas.metrics import summarization_score
from ragas import evaluate
from datasets import Dataset 


data_samples = {
    'contexts':[["A company is launching a new product, a smartphone app designed to help users track their fitness goals. The app allows users to set daily exercise targets, log their meals, and track their water intake. It also provides personalized workout recommendations and sends motivational reminders throughout the day."]],
    'summary':['A company is launching a fitness tracking app that helps users set exercise goals, log meals, and track water intake, with personalized workout suggestions and motivational reminders.'],
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[summarization_score])
score.to_pandas()
```

