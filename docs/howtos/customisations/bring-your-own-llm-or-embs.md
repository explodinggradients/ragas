# Bring Your Own LLMs and Embeddings

Ragas uses LLMs and Embeddings for both evaluation and test set generation. By default, the LLM and Embedding models of choice are OpenAI but you can easily customize the `evaluation` and `TestsetGenerator` with the LLMs and Embeddings of your choice. In this tutorial, we will go through the basics of how to do it.

:::{note}
`BaseRagasLLM` and `BaseRagasEmbeddings` are the base classes Ragas uses internally for LLMs and Embeddings. Any custom LLM or Embeddings should be a subclass of these base classes. 

If you are using Langchain, you can pass the Langchain LLM and Embeddings directly and Ragas will wrap it with `LangchainLLMWrapper` or `LangchainEmbeddingsWrapper` as needed.
:::

:::{seealso}
After understanding the basics, feel free to check out the specific guides here

- [Azure OpenAI](./azure-openai.ipynb)
- [AWS Bedrock](./aws-bedrock.ipynb)
- [Google Cloud (VertexAI)](./gcp-vertexai.ipynb)
:::

## Customizing Evaluations

Depending on which metric you use for evaluations, it will use LLMs and/or Embeddings under-the-hood. You can customize which models to use in 2 ways:

1. By Passing it through `evaluate()`: The evaluate function has 2 arguments `llm=` and `embeddings=`. You can pass any instance of `BaseRagasLLM` or `BaseRagasEmbeddings` respectively. If you are using Langchain, you can pass the Langchain llm and embeddings instances directly and Ragas will wrap it with `LangchainLLMWrapper` or `LangchainEmbeddingsWrapper` as needed.

```{code-block} python
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

langchain_llm = BaseLanguageModel(model="my_model") # any langchain LLM instance
langchain_embeddings = Embeddings(model="my_model") # any langchain Embeddings instance

results = evaluate(metrics=[], llm=langchain_llm, embeddings=embeddings)
```

2. Attaching it to `metrics`: You can attach the LLM and Embeddings to the `metrics` object directly.
```{code-block} python
# override the llm and embeddings for a specific metric
from ragas.metrics import answer_relevancy 
answer_relevancy.llm = langchain_llm
answer_relevancy.embeddings = langchain_embeddings

# You can also init a new metric with the llm and embeddings of your choice

from ragas.metrics import AnswerRelevancy
ar = AnswerRelevancy(llm=langchain_llm, embeddings=langchain_embeddings)

# pass to evaluate
result = evaluate(metrics=[ar, answer_relevancy])
# even if I pass an llm or embeddings to evaluate, it will use the ones attached to the metrics
result = evaluate(
  metrics=[ar, answer_relevancy, faithfullness], 
  llm=llm, 
  embeddings=embeddings
)
```

:::{note}
A note on precedence: llms and embeddings attached to metrics have higher precedence than the llm and embeddings passed to `evaluate()` function. You can use this to override specific metrics with a different llm or embedding models that perform better for the metric in question.
:::

## Customizing Testset Generation
There are a lot of components in the test set generation pipeline that use LLMs and Embeddings here we will be explaining the top-level components.

1. `DocumentStore`: The `DocumentStore` requires an `Extractor` to extract keywords from the documents and nodes and `embeddings` to calculate the embeddings of nodes and calculate similarity. 
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
2. `TestsetGenerator`: The `TestsetGenerator` requires `generator_llm` for evolutions, `critic_llm` for the question and node filters and `docstore` for accessing the documents. You can pass the llms when instantiating the `TestsetGenerator`.
```{code-block} python
# any langchain LLM instance
generator_llm = BaseLanguageModel(
  model="model_for_generation"
) 
# any langchain LLM instance
critic_llm = BaseLanguageModel(
  model="model_for_critic(ideally more advanced and capable)"
) 
# refer above if in doubt
docstore = InMemoryDocumentStore(
  splitter=splitter,
  embeddings=langchain_embeddings,
  extractor=keyphrase_extractor,
)
```
The `TestsetGenerator` will now init the `evolutions` and `filters` with the llms you passed. If you need further fine-grained control, you will have to initialize the `evolutions` and `filters` manually and use them. Feel free to raise an issue if the docs need clarity on this.
