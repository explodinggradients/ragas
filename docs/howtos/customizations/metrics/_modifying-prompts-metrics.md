# Modifying prompts in metrics

Every metrics in ragas that uses LLM also uses one or more prompts to come up with intermediate results that is used for formulating scores. Prompts can be treated like hyperparameters when using LLM based metrics. An optimised prompt that suits your domain and use-case can increase the accuracy of your LLM based metrics by 10-20%. An optimal prompt is also depended on the LLM one is using, so as users you might want to tune prompts that powers each metric. 

Each prompt in Ragas is written using [Prompt Object](/concepts/components/prompt/). Please make sure you have an understanding of it before going further.

### Understand the prompts of your Metric

Since Ragas treats prompts like hyperparameters in metrics, we have a unified interface of `get_prompts` to access prompts used underneath any metrics. 


```python
from ragas.metrics._simple_criteria import SimpleCriteriaScoreWithReference

scorer = SimpleCriteriaScoreWithReference(name="random", definition="some definition")
scorer.get_prompts()
```




    {'multi_turn_prompt': <ragas.metrics._simple_criteria.MultiTurnSimpleCriteriaWithReferencePrompt at 0x7f8c41410970>,
     'single_turn_prompt': <ragas.metrics._simple_criteria.SingleTurnSimpleCriteriaWithReferencePrompt at 0x7f8c41412590>}




```python
prompts = scorer.get_prompts()
print(prompts["single_turn_prompt"].to_string())
```

    Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.


### Modifying instruction in default prompt
It is highly likely that one might want to modify the prompt to suit ones needs. Ragas provides `set_prompts` methods to allow you to do so. Let's change the one of the prompts used in `FactualCorrectness` metrics


```python
prompt = scorer.get_prompts()["single_turn_prompt"]
prompt.instruction += "\nOnly output valid JSON."
```


```python
scorer.set_prompts(**{"single_turn_prompt": prompt})
```

Let's check if the prompt's instruction has actually changed


```python
print(scorer.get_prompts()["single_turn_prompt"].instruction)
```

    Given a input, system response and reference. Evaluate and score the response against the reference only using the given criteria.
    Only output valid JSON.


### Modifying examples in default prompt
Few shot examples can greatly influence the outcome of any LLM. It is highly likely that the examples in default prompt may not reflect your domain or use-case. So it's always an good practice to modify with your custom examples. Let's do one here


```python
prompt = scorer.get_prompts()["single_turn_prompt"]

prompt.examples
```




    [(SingleTurnSimpleCriteriaWithReferenceInput(user_input='Who was the director of Los Alamos Laboratory?', response='Einstein was the director of Los Alamos Laboratory.', criteria='Score responses in range of 0 (low) to 5 (high) based similarity with reference.', reference='The director of Los Alamos Laboratory was J. Robert Oppenheimer.'),
      SimpleCriteriaOutput(reason='The response and reference have two very different answers.', score=0))]




```python
from ragas.metrics._simple_criteria import (
    SingleTurnSimpleCriteriaWithReferenceInput,
    SimpleCriteriaOutput,
)
```


```python
new_example = [
    (
        SingleTurnSimpleCriteriaWithReferenceInput(
            user_input="Who was the first president of the United States?",
            response="Thomas Jefferson was the first president of the United States.",
            criteria="Score responses in range of 0 (low) to 5 (high) based similarity with reference.",
            reference="George Washington was the first president of the United States.",
        ),
        SimpleCriteriaOutput(
            reason="The response incorrectly states Thomas Jefferson instead of George Washington. While both are significant historical figures, the answer does not match the reference.",
            score=2,
        ),
    )
]
```


```python
prompt.examples = new_example
```


```python
scorer.set_prompts(**{"single_turn_prompt": prompt})
```


```python
print(scorer.get_prompts()["single_turn_prompt"].examples)
```

    [(SingleTurnSimpleCriteriaWithReferenceInput(user_input='Who was the first president of the United States?', response='Thomas Jefferson was the first president of the United States.', criteria='Score responses in range of 0 (low) to 5 (high) based similarity with reference.', reference='George Washington was the first president of the United States.'), SimpleCriteriaOutput(reason='The response incorrectly states Thomas Jefferson instead of George Washington. While both are significant historical figures, the answer does not match the reference.', score=2))]


Let's now view and verify the full new prompt with modified instruction and examples


```python
scorer.get_prompts()["single_turn_prompt"].to_string()
```
