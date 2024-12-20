# Caching in Ragas

You can use caching to speed up your evaluations and testset generation by avoiding redundant computations. We use Exact Match Caching to cache the responses from the LLM and Embedding models.

You can use the [DiskCacheBackend][ragas.cache.DiskCacheBackend] which uses a local disk cache to store the cached responses. You can also implement your own custom cacher by implementing the [CacheInterface][ragas.cache.CacheInterface].


## Using DefaultCacher

Let's see how you can use the [DiskCacheBackend][ragas.cache.DiskCacheBackend]  LLM and Embedding models.



```python
from ragas.cache import DiskCacheBackend

cacher = DiskCacheBackend()
cacher
```




    DiskCacheBackend(cache_dir=.cache)



Create an LLM and Embedding model with the cacher, here I'm using the [ChatOpenAI][langchain_openai.ChatOpenAI] model from [langchain-openai](https://github.com/langchain-ai/langchain-openai) as an example.



```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

cached_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"), cache=cacher)
```

Now let's run a simple evaluation.


```python
from ragas import evaluate
from ragas import EvaluationDataset

from ragas.metrics import FactualCorrectness
from datasets import load_dataset

# load the dataset
dataset = load_dataset(
    "explodinggradients/amnesty_qa",
    "english_v3",
    trust_remote_code=True
)
eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])

# evaluate the dataset
results = evaluate(
    dataset=eval_dataset,
    metrics=[FactualCorrectness(llm=cached_llm)],
)

results

```

    Repo card metadata block was not found. Setting CardData to empty.



    Evaluating:   0%|          | 0/20 [00:00<?, ?it/s]





    {'factual_correctness': 0.3465}



This took 54 seconds to run in our local machine. Now let's run it again to see the cache in action.



```python
results = evaluate(
    dataset=eval_dataset, metrics=[FactualCorrectness(llm=cached_llm)],
)

results
```


    Evaluating:   0%|          | 0/20 [00:00<?, ?it/s]





    {'factual_correctness': 0.3365}



Runs almost instantaneously.

You can also use this with testset generation also by replacing the `generator_llm` with a cached version of it. Refer to the [testset generation](../../getstarted/rag_testset_generation.md) section for more details.
