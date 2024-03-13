# Bring Your Own LLMs and Embeddings

Ragas uses LLMs and Embeddings for both evaluation and test set generation. By default, the LLM and Embedding models of choice are OpenAI models.

- [Evaluations](#evaluations)
- [Testset Generation](#test-set-generation)


:::{note}
`BaseRagasLLM` and `BaseRagasEmbeddings` are the base classes Ragas uses internally for LLMs and Embeddings. Any custom LLM or Embeddings should be a subclass of these base classes. 

If you are using Langchain, you can pass the Langchain LLM and Embeddings directly and Ragas will wrap it with `LangchainLLMWrapper` or `LangchainEmbeddingsWrapper` as needed.
:::

:::{seealso}
After understanding the basics, feel free to check out the specific guides here

- [Azure OpenAI](./azure-openai.ipynb)
- [AWS Bedrock](./aws-bedrock.ipynb)
- [Google Cloud (VertexAI)](./gcp-vertexai.ipynb)

Checkout list of embeddings supported by langchain [here](https://python.langchain.com/docs/integrations/text_embedding/)
Checkout list of llms supported by langchain [here](https://python.langchain.com/docs/integrations/chat/)

:::

For the sake of this example I will be using `m2-bert-80M-8k-retrieval` to replace the default OpenAI Embeddings, and `NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT` hosted on Together AI as LLM to replace the default gpt-3.5-turbo LLM. So let's start by initializing them.
  
```{code-block} python

from langchain_together import Together
from langchain_together.embeddings import TogetherEmbeddings

together_key = "<your-key-here>"

embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

together_completion = Together(
    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
    temperature=0.7,
    max_tokens=4000,
    top_k=1,
    together_api_key=together_key
)
```


## Evaluations

Depending on which metric you use for evaluations, it will use LLMs and/or Embeddings under-the-hood. You can customize which models to use in 2 ways:

1. By Passing it through `evaluate()`: The evaluate function has 2 arguments `llm` and `embeddings`. You can pass any instance of `BaseRagasLLM` or `BaseRagasEmbeddings` respectively. If you are using Langchain, you can pass the Langchain llm and embeddings instances directly and Ragas will wrap it with `LangchainLLMWrapper` or `LangchainEmbeddingsWrapper` as needed.

```{code-block} python
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

langchain_llm =  # any langchain LLM instance
langchain_embeddings = # any langchain Embeddings instance

results = evaluate(metrics=[], llm=langchain_llm, embeddings=embeddings)
```

For example, to use the embeddings and llm we initialized above, we can do the following:
```{code-block} python

from ragas.metrics import faithfullness
from ragas import evaluate

results = evaluate(metrics=[faithfullness], llm=together_completion, embeddings=embeddings)
```

2. Attaching it to `metrics`: You can attach the LLM and Embeddings to the `metrics` object directly.
This method can be more useful when you want a specific metric to use a different LLM or Embeddings than the rest of the metrics.

```{code-block} python
# override the llm and embeddings for a specific metric
from ragas.metrics import answer_relevancy 

answer_relevancy.llm = langchain_llm
answer_relevancy.embeddings = langchain_embeddings

# You can also init a new metric with the llm and embeddings of your choice

from ragas.metrics import AnswerRelevancy
answer_relevancy_duplicate = AnswerRelevancy(llm=langchain_llm, embeddings=langchain_embeddings)

# pass to evaluate
result = evaluate(metrics=[answer_relevancy_duplicate, answer_relevancy])
result = evaluate(
  metrics=[answer_relevancy_duplicate, answer_relevancy,], 
)
```

For example, to use the embeddings and llm we initialized above, we can do the following:

```{code-block} python
from ragas.metrics import answer_relevancy 

answer_relevancy.llm = together_completion
answer_relevancy.embeddings = together_embeddings

result = evaluate(metrics=[answer_relevancy])
```

:::{note}
A note on precedence: llms and embeddings attached to metrics have higher precedence than the llm and embeddings passed to `evaluate()` function. You can use this to override specific metrics with a different llm or embedding models that perform better for the metric in question.
:::

## Test set Generation


There are a lot of components in the test set generation pipeline that use LLMs and Embeddings here we will be explaining the top-level components.

- [LLMs and Embeddings](#llms)
- [DocumentStore](#documentstore)


### LLMs
There are two type of LLMs used in the test set generation pipeline. The `generator_llm` and `critic_llm`. The `generator_llm` is the component that generates the questions, and evolves the question to make it more relevant. The `critic_llm` is the component that filters the questions and nodes based on the question and node relevance. Both uses OpenAI models by default. To replace them with your own LLMs, you can pass the llms when instantiating the `TestsetGenerator`.

It also uses `embeddings` for functionalities like calculating the similarity between nodes,etc.

```{code-block} python
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

# define llm and embeddings
langchain_llm = BaseLanguageModel(model="my_model") # any langchain LLM instance
langchain_embeddings = Embeddings(model="my_model") # any langchain Embeddings instance

# make sure to wrap them with wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

langchain_llm = LangchainLLMWrapper(langchain_llm)
langchain_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

# you can also use custom LLMs and Embeddings here but make sure 
# they are subclasses of BaseRagasLLM and BaseRagasEmbeddings
llm = MyCustomLLM()
embeddings = MyCustomEmbeddings()
```

For example, let's replace the default OpenAI LLM and Embeddings with the ones we initialized above.

```{code-block} python

from ragas.testset.generator import TestsetGenerator


# generator with custom llm and embeddings
generator = TestsetGenerator.from_langchain(
    generator_llm=together_completion,
    critic_llm=together_completion,
    embeddings=together_embeddings,
) 

```

### DocumentStore
The `DocumentStore` requires an `Extractor` to extract keywords from the documents and nodes and `embeddings` to calculate the embeddings of nodes and calculate similarity. 

```{code-block} python
# default extractor
from ragas.testset.extractor import KeyphraseExtractor
from langchain.text_splitter import TokenTextSplitter
# default DocumentStore
from ragas.testset.docstore import InMemoryDocumentStore

# define llm and embeddings
langchain_llm = BaseLanguageModel(model="my_model") # any langchain LLM instance
langchain_embeddings = Embeddings(model="my_model") # any langchain Embeddings instance
# make sure to wrap them with wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
langchain_llm = LangchainLLMWrapper(langchain_llm)
langchain_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

# you can also use custom LLMs and Embeddings here but make sure 
# they are subclasses of BaseRagasLLM and BaseRagasEmbeddings
llm = MyCustomLLM()
embeddings = MyCustomEmbeddings()

# init the DocumentStore with your own llm and embeddings
splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
keyphrase_extractor = KeyphraseExtractor(llm=langchain_llm)
docstore = InMemoryDocumentStore(
    splitter=splitter,
    embeddings=langchain_embeddings,
    extractor=keyphrase_extractor,
)
```


The `TestsetGenerator` will now init the `evolutions` and `filters` with the llms you passed. If you need further fine-grained control, you will have to initialize the `evolutions` and `filters` manually and use them. Feel free to raise an issue if the docs need clarity on this.
