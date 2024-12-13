# Customize timeouts and rate limits

The `RunConfig` allows you to pass in the run parameters to functions like `evaluate()` and `TestsetGenerator.generate()`. Depending on your LLM providers rate limits, SLAs and traffic, controlling these parameters can improve the speed and reliability and set failure tolerance of Ragas runs.

How to configure the `RunConfig` in

- [Evaluate](#evaluate)
- [TestsetGenerator]()

## Rate Limits

Ragas leverages parallelism with Async in python but the `RunConfig` has a field called `max_workers` which control the number of concurent requests allowed together. You adjust this to get the maximum concurency your provider allows


```python
from ragas.run_config import RunConfig

# increasing max_workers to 64 and timeout to 60 seconds

my_run_config = RunConfig(max_workers=64, timeout=60)
```

### Evaluate


```python
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness
from datasets import load_dataset
from ragas import evaluate

dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3")

samples = []
for row in dataset["eval"]:
    sample = SingleTurnSample(
        user_input=row["user_input"],
        reference=row["reference"],
        response=row["response"],
        retrieved_contexts=row["retrieved_contexts"],
    )
    samples.append(sample)

eval_dataset = EvaluationDataset(samples=samples)
metric = Faithfulness()

_ = evaluate(
    dataset=eval_dataset,
    metrics=[metric],
    run_config=my_run_config,
)
```
