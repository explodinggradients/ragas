# LlamaIndex

[LlamaIndex](https://github.com/run-llama/llama_index) is a data framework for LLM applications to ingest, structure, and access private or domain-specific data. Makes it super easy to connect LLMs with your own data. But in order to figure out the best configuration for llamaIndex and your data you need a object measure of the performance. This is where ragas comes in. Ragas will help you evaluate your `QueryEngine` and gives you the confidence to tweak the configuration to get hightest score.

This guide assumes you have familarity with the LlamaIndex framework.

## Building the Testset

You will need an testset to evaluate your `QueryEngine` against. You can either build one yourself or use the [Testset Generator Module](../../getstarted/testset_generation.md) in Ragas to get started with a small synthetic one.

Let's see how that works with Llamaindex


```python
# load the documents
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./nyc_wikipedia").load_data()
```

Now  lets init the `TestsetGenerator` object with the corresponding generator and critic llms


```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# generator with openai models
generator_llm = OpenAI(model="gpt-3.5-turbo-16k")
critic_llm = OpenAI(model="gpt-4")
embeddings = OpenAIEmbedding()

generator = TestsetGenerator.from_llama_index(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings,
)
```

Now you are all set to generate the dataset


```python
# generate testset
testset = generator.generate_with_llamaindex_docs(
    documents,
    test_size=5,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)
```


    embedding nodes:   0%|          | 0/54 [00:00<?, ?it/s]


    Filename and doc_id are the same for all nodes.



    Generating:   0%|          | 0/5 [00:00<?, ?it/s]



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
      <th>question</th>
      <th>contexts</th>
      <th>ground_truth</th>
      <th>evolution_type</th>
      <th>metadata</th>
      <th>episode_done</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What cultural movement began in New York City ...</td>
      <td>[ Others cite the end of the crack epidemic an...</td>
      <td>The Harlem Renaissance</td>
      <td>simple</td>
      <td>[{'file_path': '/home/jjmachan/jjmachan/explod...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is the significance of New York City's tr...</td>
      <td>[ consisting of 51 council members whose distr...</td>
      <td>New York City's transportation system is both ...</td>
      <td>simple</td>
      <td>[{'file_path': '/home/jjmachan/jjmachan/explod...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What factors led to the creation of Central Pa...</td>
      <td>[ next ten years with British troops stationed...</td>
      <td>Public-minded members of the contemporaneous b...</td>
      <td>reasoning</td>
      <td>[{'file_path': '/home/jjmachan/jjmachan/explod...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What was the impact of the Treaty of Breda on ...</td>
      <td>[ British raids. In 1626, the Dutch colonial D...</td>
      <td>The Treaty of Breda confirmed the transfer of ...</td>
      <td>multi_context</td>
      <td>[{'file_path': '/home/jjmachan/jjmachan/explod...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What role did New York play in the American Re...</td>
      <td>[ British raids. In 1626, the Dutch colonial D...</td>
      <td>New York played a significant role in the Amer...</td>
      <td>simple</td>
      <td>[{'file_path': '/home/jjmachan/jjmachan/explod...</td>
      <td>True</td>
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
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings

vector_index = VectorStoreIndex.from_documents(documents)

query_engine = vector_index.as_query_engine()
```

Lets try an sample question from the generated testset to see if it is working


```python
# convert it to pandas dataset
df = testset.to_pandas()
df["question"][0]
```




    'What cultural movement began in New York City and established the African-American literary canon in the United States?'




```python
response_vector = query_engine.query(df["question"][0])

print(response_vector)
```

    The Harlem Renaissance was the cultural movement that began in New York City and established the African-American literary canon in the United States.


## Evaluating the `QueryEngine`

Now that we have a `QueryEngine` for the `VectorStoreIndex` we can use the llama_index integration Ragas has to evaluate it. 

In order to run an evaluation with Ragas and LlamaIndex you need 3 things

1. LlamaIndex `QueryEngine`: what we will be evaluating
2. Metrics: Ragas defines a set of metrics that can measure different aspects of the `QueryEngine`. The available metrics and their meaning can be found [here](https://github.com/explodinggradients/ragas/blob/main/docs/metrics.md)
3. Questions: A list of questions that ragas will test the `QueryEngine` against. 

first lets generate the questions. Ideally you should use that you see in production so that the distribution of question with which we evaluate matches the distribution of questions seen in production. This ensures that the scores reflect the performance seen in production but to start off we'll be using a few example question.

Now lets import the metrics we will be using to evaluate


```python
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.metrics.critique import harmfulness

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    harmfulness,
]
```

now lets init the evaluator model


```python
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# using GPT 3.5, use GPT 4 / 4-turbo for better accuracy
evaluator_llm = OpenAI(model="gpt-3.5-turbo")
```

the `evaluate()` function expects a dict of "question" and "ground_truth" for metrics. You can easily convert the `testset` to that format


```python
# convert to HF dataset
ds = testset.to_dataset()

ds_dict = ds.to_dict()
ds_dict["question"]
ds_dict["ground_truth"]
```




    ['The Harlem Renaissance',
     "New York City's transportation system is both complex and extensive, with a comprehensive mass transit system that accounts for one in every three users of mass transit in the United States. The New York City Subway system is the largest rapid transit system in the world, and the city has a high usage of public transport, with a majority of households not owning a car. Due to their reliance on mass transit, New Yorkers spend less of their household income on transportation compared to the national average.",
     'Public-minded members of the contemporaneous business elite lobbied for the establishment of Central Park',
     'The Treaty of Breda confirmed the transfer of New Amsterdam to English control and the renaming of the settlement as New York. The Duke of York, who would later become King James II and VII, played a significant role in the naming of New York City.',
     'New York played a significant role in the American Revolution. The Stamp Act Congress met in New York in October 1765, and the city became a center for the Sons of Liberty organization. Skirmishes and battles took place in and around New York, including the Battle of Long Island and the Battle of Saratoga. The city was occupied by British forces for much of the war, but it was eventually liberated by American troops in 1783.']



Finally lets run the evaluation


```python
from ragas.integrations.llama_index import evaluate

result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=ds_dict,
    llm=evaluator_llm,
    embeddings=OpenAIEmbedding(),
)
```


    Running Query Engine:   0%|          | 0/5 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/25 [00:00<?, ?it/s]


    n values greater than 1 not support for LlamaIndex LLMs
    n values greater than 1 not support for LlamaIndex LLMs
    n values greater than 1 not support for LlamaIndex LLMs
    n values greater than 1 not support for LlamaIndex LLMs
    n values greater than 1 not support for LlamaIndex LLMs



```python
# final scores
print(result)
```

    {'faithfulness': 0.9000, 'answer_relevancy': 0.8993, 'context_precision': 0.9000, 'context_recall': 1.0000, 'harmfulness': 0.0000}


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
      <th>question</th>
      <th>contexts</th>
      <th>answer</th>
      <th>ground_truth</th>
      <th>faithfulness</th>
      <th>answer_relevancy</th>
      <th>context_precision</th>
      <th>context_recall</th>
      <th>harmfulness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What cultural movement began in New York City ...</td>
      <td>[=== 19th century ===\n\nOver the course of th...</td>
      <td>The Harlem Renaissance of literary and cultura...</td>
      <td>The Harlem Renaissance</td>
      <td>0.5</td>
      <td>0.907646</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is the significance of New York City's tr...</td>
      <td>[== Transportation ==\n\nNew York City's compr...</td>
      <td>New York City's transportation system is signi...</td>
      <td>New York City's transportation system is both ...</td>
      <td>1.0</td>
      <td>0.986921</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What factors led to the creation of Central Pa...</td>
      <td>[=== 19th century ===\n\nOver the course of th...</td>
      <td>Prominent American literary figures lived in N...</td>
      <td>Public-minded members of the contemporaneous b...</td>
      <td>1.0</td>
      <td>0.805014</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What was the impact of the Treaty of Breda on ...</td>
      <td>[=== Dutch rule ===\n\nA permanent European pr...</td>
      <td>The Treaty of Breda resulted in the transfer o...</td>
      <td>The Treaty of Breda confirmed the transfer of ...</td>
      <td>1.0</td>
      <td>0.860931</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What role did New York play in the American Re...</td>
      <td>[=== Province of New York and slavery ===\n\nI...</td>
      <td>New York served as a significant location duri...</td>
      <td>New York played a significant role in the Amer...</td>
      <td>1.0</td>
      <td>0.935846</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


