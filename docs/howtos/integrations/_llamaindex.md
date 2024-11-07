# LlamaIndex

[LlamaIndex](https://github.com/run-llama/llama_index) is a data framework for LLM applications to ingest, structure, and access private or domain-specific data. Makes it super easy to connect LLMs with your own data. But in order to figure out the best configuration for llamaIndex and your data you need a object measure of the performance. This is where ragas comes in. Ragas will help you evaluate your `QueryEngine` and gives you the confidence to tweak the configuration to get hightest score.

This guide assumes you have familarity with the LlamaIndex framework.

## Building the Testset

You will need an testset to evaluate your `QueryEngine` against. You can either build one yourself or use the [Testset Generator Module](../../getstarted/testset_generation.md) in Ragas to get started with a small synthetic one.

Let's see how that works with Llamaindex

# load the documents
```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./nyc_wikipedia").load_data()
```

Now lets init the `TestsetGenerator` object with the corresponding generator and critic llms


```python
from ragas.testset import TestsetGenerator

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# generator with openai models
generator_llm = OpenAI(model="gpt-4o")
embeddings = OpenAIEmbedding(model="text-embedding-3-large")

generator = TestsetGenerator.from_llama_index(
    llm=generator_llm,
    embedding_model=embeddings,
)
```

Now you are all set to generate the dataset


```python
# generate testset
testset = generator.generate_with_llamaindex_docs(
    documents,
    testset_size=5,
)
```


```python
df = testset.to_pandas()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>reference_contexts</th>
      <th>reference</th>
      <th>synthesizer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Why was New York named after the Duke of York?</td>
      <td>[Etymology ==\n\nIn 1664, New York was named i...</td>
      <td>New York was named after the Duke of York in 1...</td>
      <td>AbstractQuerySynthesizer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>How did the early Europan exploraton and setle...</td>
      <td>[History ==\n\n\n=== Early history ===\nIn the...</td>
      <td>The early European exploration and settlement ...</td>
      <td>AbstractQuerySynthesizer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>New York City population culture finance diver...</td>
      <td>[New York City, the most populous city in the ...</td>
      <td>New York City is a global cultural, financial,...</td>
      <td>ComparativeAbstractQuerySynthesizer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How do the economic aspects of New York City, ...</td>
      <td>[New York City, the most populous city in the ...</td>
      <td>New York City's economic aspects, such as its ...</td>
      <td>ComparativeAbstractQuerySynthesizer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What role do biomedical research institutions ...</td>
      <td>[Education ==\n\n \n\nNew York City has the la...</td>
      <td>Biomedical research institutions in New York C...</td>
      <td>SpecificQuerySynthesizer</td>
    </tr>
  </tbody>
</table>
</div>



with a test dataset to test our `QueryEngine` lets now build one and evaluate it.

## Building the `QueryEngine`

To start lets build an `VectorStoreIndex` over the New York Citie's [wikipedia page](https://en.wikipedia.org/wiki/New_York_City) as an example and use ragas to evaluate it. 

Since we already loaded the dataset into `documents` lets use that.


```python
# build query engine
from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex.from_documents(documents)

query_engine = vector_index.as_query_engine()
```

Lets try an sample question from the generated testset to see if it is working


```python
# convert it to pandas dataset
df = testset.to_pandas()
df["user_input"][0]
```




    'Why was New York named after the Duke of York?'




```python
response_vector = query_engine.query(df["user_input"][0])

print(response_vector)
```

    New York was named after the Duke of York because in 1664, the city was named in honor of the Duke of York, who later became King James II of England.


## Evaluating the `QueryEngine`

Now that we have a `QueryEngine` for the `VectorStoreIndex` we can use the llama_index integration Ragas has to evaluate it. 

In order to run an evaluation with Ragas and LlamaIndex you need 3 things

1. LlamaIndex `QueryEngine`: what we will be evaluating
2. Metrics: Ragas defines a set of metrics that can measure different aspects of the `QueryEngine`. The available metrics and their meaning can be found [here](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
3. Questions: A list of questions that ragas will test the `QueryEngine` against. 

first lets generate the questions. Ideally you should use that you see in production so that the distribution of question with which we evaluate matches the distribution of questions seen in production. This ensures that the scores reflect the performance seen in production but to start off we'll be using a few example question.

Now lets import the metrics we will be using to evaluate


```python
# import metrics
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# init metrics with evaluator LLM
from ragas.llms import LlamaIndexLLMWrapper

evaluator_llm = LlamaIndexLLMWrapper(OpenAI(model="gpt-4o"))
metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm),
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
]
```

the `evaluate()` function expects a dict of "question" and "ground_truth" for metrics. You can easily convert the `testset` to that format


```python
# convert to Ragas Evaluation Dataset
ragas_dataset = testset.to_evaluation_dataset()
ragas_dataset
```




    EvaluationDataset(features=['user_input', 'reference_contexts', 'reference'], len=7)



Finally lets run the evaluation


```python
from ragas.integrations.llama_index import evaluate

result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=ragas_dataset,
)
```


```python
# final scores
print(result)
```

    {'faithfulness': 0.9746, 'answer_relevancy': 0.9421, 'context_precision': 0.9286, 'context_recall': 0.6857}


You can convert into a pandas dataframe to run more analysis on it.


```python
result.to_pandas()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>reference_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>faithfulness</th>
      <th>answer_relevancy</th>
      <th>context_precision</th>
      <th>context_recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What events led to New York being named after ...</td>
      <td>[New York City is the headquarters of the glob...</td>
      <td>[Etymology ==\n\nIn 1664, New York was named i...</td>
      <td>New York was named in honor of the Duke of Yor...</td>
      <td>New York was named after the Duke of York in 1...</td>
      <td>1.000000</td>
      <td>0.950377</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>How early European explorers and Native Americ...</td>
      <td>[=== Dutch rule ===\n\nA permanent European pr...</td>
      <td>[History ==\n\n\n=== Early history ===\nIn the...</td>
      <td>Early European explorers established a permane...</td>
      <td>Early European explorers and Native Americans ...</td>
      <td>1.000000</td>
      <td>0.896300</td>
      <td>1.0</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>New York City population economy challenges</td>
      <td>[=== Wealth and income disparity ===\nNew York...</td>
      <td>[New York City, the most populous city in the ...</td>
      <td>New York City has faced challenges related to ...</td>
      <td>New York City, as the most populous city in th...</td>
      <td>1.000000</td>
      <td>0.915717</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How do the economic aspects of New York City, ...</td>
      <td>[=== Wealth and income disparity ===\nNew York...</td>
      <td>[New York City, the most populous city in the ...</td>
      <td>The economic aspects of New York City, as a gl...</td>
      <td>New York City's economic aspects as a global c...</td>
      <td>0.913043</td>
      <td>0.929317</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What are some of the cultural and architectura...</td>
      <td>[==== Staten Island ====\nStaten Island (Richm...</td>
      <td>[Geography ==\n\nDuring the Wisconsin glaciati...</td>
      <td>Brooklyn is known for its cultural diversity, ...</td>
      <td>Brooklyn is distinct within New York City due ...</td>
      <td>1.000000</td>
      <td>0.902664</td>
      <td>0.5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>What measures has New York City implemented to...</td>
      <td>[==== International events ====\nIn terms of h...</td>
      <td>[Environment ==\n\n \nEnvironmental issues in ...</td>
      <td>New York City has implemented various measures...</td>
      <td>New York City has implemented several measures...</td>
      <td>0.909091</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>What role did New York City play during the Am...</td>
      <td>[=== Province of New York and slavery ===\n\nI...</td>
      <td>[History ==\n\n\n=== Early history ===\nIn the...</td>
      <td>New York City served as a significant military...</td>
      <td>During the American Revolution, New York City ...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


