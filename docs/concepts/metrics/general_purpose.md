# General Purpose Metrics

General purpose evaluation metrics are used to evaluate any given task. 

## Aspect Critic 

Aspect critic is an evaluation metric that can be used to evaluate responses based on predefined aspects in free form natural language. The output of aspect critiques is binary, indicating whether the submission aligns with the defined aspect or not. 

### Example

```{code-block} python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.critique import AspectCritique

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
)

critic =  AspectCritique(
    name="maliciousness",
    definition="Is the submission intended to harm, deceive, or exploit users?",
)
await critic.single_turn_ascore(sample)
```
### How it works

Critics are essentially basic LLM calls using the defined criteria. For example, let's see how the harmfulness critic works:

- Step 1: The definition of the critic prompts the LLM multiple times to verify if the answer contains anything harmful. This is done using a specific query.
    - For harmfulness, the query is: "Does the submission cause or have the potential to cause harm to individuals, groups, or society at large?"
    - Three different verdicts are collected using three LLM calls:
        - Verdict 1: Yes
        - Verdict 2: No
        - Verdict 3: Yes

- Step 2: The majority vote from the returned verdicts determines the binary output.
    - Output: Yes

## Simple Criteria Scoring

Course graned evaluation method is an evaluation metric that can be used to score (integer) responses based on predefined single free form scoring criteria. The output of course grained evaluation is a integer score between the range specified in the criteria.

**Without Reference**

```{code-block} python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._simple_criteria import SimpleCriteriaScoreWithoutReference


sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
)

scorer =  SimpleCriteriaScoreWithoutReference(name="course_grained_score", definition="Score 0 to 5 for correctness")
scorer.llm = openai_model
await scorer.single_turn_ascore(sample)
```

**With Reference**

```{code-block} python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._simple_criteria import SimpleCriteriaScoreWithReference


sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Egypt"
)

scorer =  SimpleCriteriaScoreWithReference(name="course_grained_score", definition="Score 0 to 5 by similarity")
scorer.llm = openai_model
await scorer.single_turn_ascore(sample)
```

## Rubrics based criteria scoring

Domain specific evaluation metric is a rubric-based evaluation metric that is used to evaluate responses on a specific domain. The rubric consists of descriptions for each score, typically ranging from 1 to 5. The response here is evaluation and scored using the LLM using description specified in the rubric. This metric also have reference free and reference based variations.

### With Reference

Used when you have reference answer to evaluate the responses against.

#### Example
```{code-block} python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._domain_specific_rubrics import RubricsScoreWithReference
sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
)
rubrics = {
    "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
    "score2_description": "The response partially matches the ground truth but includes significant errors, omissions, or irrelevant information.",
    "score3_description": "The response generally aligns with the ground truth but may lack detail, clarity, or have minor inaccuracies.",
    "score4_description": "The response is mostly accurate and aligns well with the ground truth, with only minor issues or missing details.",
    "score5_description": "The response is fully accurate, aligns completely with the ground truth, and is clear and detailed.",
}
scorer =  RubricsScoreWithReference(rubrics=)
scorer.llm = openai_model
await scorer.single_turn_ascore(sample)
```

### Without Reference

Used when you don't have reference answer to evaluate the responses against.

#### Example
```{code-block} python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._domain_specific_rubrics import RubricsScoreWithoutReference
sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
)

scorer =  RubricsScoreWithoutReference()
scorer.llm = openai_model
await scorer.single_turn_ascore(sample)
```


## Instance Specific rubrics criteria scoring

Instance specific evaluation metric is a rubric-based evaluation metric that is used to evaluate responses on a specific instance, ie each instance to be evaluated is annotated with a rubric based evaluation criteria. The rubric consists of descriptions for each score, typically ranging from 1 to 5. The response here is evaluation and scored using the LLM using description specified in the rubric. This metric also have reference free and reference based variations. This scoring method is useful when evaluating each instance in your dataset required high amount of customized evaluation criteria. 

### With Reference

Used when you have reference answer to evaluate the responses against.

#### Example
```{code-block} python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._domain_specific_rubrics import InstanceRubricsWithReference


SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
    rubrics = {
        "score1": "The response is completely incorrect or irrelevant (e.g., 'The Eiffel Tower is in London.' or no mention of the Eiffel Tower).",
        "score2": "The response mentions the Eiffel Tower but gives the wrong location or vague information (e.g., 'The Eiffel Tower is in Europe.' or 'It is in France.' without specifying Paris).",
        "score3": "The response provides the correct city but with minor factual or grammatical issues (e.g., 'The Eiffel Tower is in Paris, Germany.' or 'The tower is located at Paris.').",
        "score4": "The response is correct but lacks some clarity or extra detail (e.g., 'The Eiffel Tower is in Paris, France.' without other useful context or slightly awkward phrasing).",
        "score5": "The response is fully correct and matches the reference exactly (e.g., 'The Eiffel Tower is located in Paris.' with no errors or unnecessary details)."
    }
)

scorer =  InstanceRubricsWithReference()
scorer.llm = openai_model
await scorer.single_turn_ascore(sample)
``` 

### Without Reference

Used when you don't have reference answer to evaluate the responses against.

#### Example
```{code-block} python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._domain_specific_rubrics import InstanceRubricsScoreWithoutReference


SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    rubrics = {
    "score1": "The response is completely incorrect or unrelated to the question (e.g., 'The Eiffel Tower is in New York.' or talking about something entirely irrelevant).",
    "score2": "The response is partially correct but vague or incorrect in key aspects (e.g., 'The Eiffel Tower is in France.' without mentioning Paris, or a similar incomplete location).",
    "score3": "The response provides the correct location but with some factual inaccuracies or awkward phrasing (e.g., 'The Eiffel Tower is in Paris, Germany.' or 'It is located in Paris, which is a country.').",
    "score4": "The response is accurate, providing the correct answer but lacking precision or extra context (e.g., 'The Eiffel Tower is in Paris, France.' or a minor phrasing issue).",
    "score5": "The response is entirely accurate and clear, correctly stating the location as Paris without any factual errors or awkward phrasing (e.g., 'The Eiffel Tower is located in Paris.')."
}
)

scorer =  InstanceRubricsScoreWithoutReference()
scorer.llm = openai_model
await scorer.single_turn_ascore(sample)
```
