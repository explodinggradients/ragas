# Adapting metrics to target language

While using ragas to evaluate LLM application workflows, you may have applications to be evaluated that are in languages other than english. In this case, it is best to adapt your LLM powered evaluation metrics to the target language. One obivous way to do this is to manually change the instruction and demonstration, but this can be time consuming. Ragas here offers automatic language adaptation where you can automatically adapt any metrics to target language by using LLM itself. This notebook demonstrates this with simple example

For the sake of this example, let's choose and metric and inspect the default prompts


```python
from ragas.metrics import SimpleCriteriaScoreWithReference

scorer = SimpleCriteriaScoreWithReference(
    name="course_grained_score", definition="Score 0 to 5 by similarity"
)
```


```python
scorer.get_prompts()
```




    {'multi_turn_prompt': <ragas.metrics._simple_criteria.MultiTurnSimpleCriteriaWithReferencePrompt at 0x7fcf409c3880>,
     'single_turn_prompt': <ragas.metrics._simple_criteria.SingleTurnSimpleCriteriaWithReferencePrompt at 0x7fcf409c3a00>}



As you can see, the instruction and demonstration are both in english. Setting up LLM to be used for this conversion


```python
from ragas.llms import llm_factory

llm = llm_factory()
```

To view the supported language codes


```python
from ragas.utils import RAGAS_SUPPORTED_LANGUAGE_CODES

print(list(RAGAS_SUPPORTED_LANGUAGE_CODES.keys()))
```

    ['english', 'hindi', 'marathi', 'chinese', 'spanish', 'amharic', 'arabic', 'armenian', 'bulgarian', 'urdu', 'russian', 'polish', 'persian', 'dutch', 'danish', 'french', 'burmese', 'greek', 'italian', 'japanese', 'deutsch', 'kazakh', 'slovak']


Now let's adapt it to 'hindi' as the target language using `adapt` method.
Language adaptation in Ragas works by translating few shot examples given along with the prompts to the target language. Instructions remains in english. 


```python
adapted_prompts = await scorer.adapt_prompts(language="hindi", llm=llm)
```

Inspect the adapted prompts and make corrections if needed


```python
adapted_prompts
```




    {'multi_turn_prompt': <ragas.metrics._simple_criteria.MultiTurnSimpleCriteriaWithReferencePrompt at 0x7fcf42bc40a0>,
     'single_turn_prompt': <ragas.metrics._simple_criteria.SingleTurnSimpleCriteriaWithReferencePrompt at 0x7fcf722de890>}



set the prompts to new adapted prompts using `set_prompts` method


```python
scorer.set_prompts(**adapted_prompts)
```

Evaluate using adapted metrics


```python
from ragas.dataset_schema import SingleTurnSample

sample = SingleTurnSample(
    user_input="एफिल टॉवर कहाँ स्थित है?",
    response="एफिल टॉवर पेरिस में स्थित है।",
    reference="एफिल टॉवर मिस्र में स्थित है",
)

scorer.llm = llm
await scorer.single_turn_ascore(sample)
```




    0



Trace of reasoning and score

`{
    "reason": "प्रतिक्रिया और संदर्भ के उत्तर में स्थान के संदर्भ में महत्वपूर्ण भिन्नता है।",
    "score": 0
}`


