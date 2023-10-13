(core-concepts)=
# Core Concepts
:::{toctree}
:caption: Concepts
:hidden:

metrics_driven
metrics/index
testset_generation
feedback
:::

Ragas aims to create an open standard, providing devs with the tools and techniques to leverage continual learning in their RAG applications. With Ragas, you would be able to

1. Synthetically generate a diverse test dataset that you can use to evaluate your app.
2. use advanced metrics we built to measure how your app performs.
3. help monitor your apps in production with custom models and see if there are any discrepancies.
4. bring up those discrepancies and build new datasets so that you can test and refine your app further to solve them.

(what-is-rag)=
:::{dropdown} what is RAG and continual learning?
```{rubric} RAG
```

Retrieval Augmented Generation (RAG) is a natural language processing (NLP) technique that combines the strengths of retrieval- and generative-based artificial intelligence (AI) models. 
 RAG uses an information retrieval system to provide data to a Large Language Model (LLM). 
 RAG models first use a retriever to identify a set of relevant documents from a knowledge base. 
 RAG can provide more accurate results to queries than a generative LLM on its own because RAG uses knowledge external to data already contained in the LLM.

```{rubric} Continual Learning
```

With continual learning, models continuously learn and evolve based on the input of increasing amounts of data while retaining previously-learned knowledge. The goal is to develop autonomous agents that can learn continuously and adaptively to develop skills necessary for performing more complex tasks without forgetting what has been learned before. 

The goal of continual learning is to: 
- Use data that is coming in to automatically retrain the model
- Gain high accuracy and retain high performing models
:::

::::{grid} 2

:::{grid-item-card} Metrics Driven Development
:link: mdd
:link-type: ref
What is MDD?
:::

:::{grid-item-card} Ragas Metrics
:link: ragas-metrics
:link-type: ref
What metrics are available? How do they work?
:::

:::{grid-item-card} Synthetic Test Data Generation
:link: testset-generation
:link-type: ref
How to create more datasets to test on?
:::

:::{grid-item-card} Utilizing User Feedback
:link: user-feedback
:link-type: ref
How to leverage the signals from user to improve?
:::
::::
