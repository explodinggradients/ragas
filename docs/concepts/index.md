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

Ragas aims to create an open standard, providing developers with the tools and techniques to leverage continual learning in their RAG applications. With Ragas, you would be able to

1. Synthetically generate a diverse test dataset that you can use to evaluate your app.
2. Use LLM-assisted evaluation metrics designed to help you objectively measure the performance of your application.
3. Monitor the quality of your apps in production using smaller, cheaper models that can give actionable insights. For example, the number of hallucinations in the generated answer. 
4. Use these insights to iterate and improve your application.


(what-is-rag)=
:::{dropdown} what is RAG and continual learning?
```{rubric} RAG
```

Retrieval augmented generation (RAG) is a paradigm for augmenting LLM with custom data. It generally consists of two stages:

- indexing stage: preparing a knowledge base, and

- querying stage: retrieving relevant context from the knowledge to assist the LLM in responding to a question

```{rubric} Continual Learning
```

Continual learning is concept used in machine learning that aims to learn, iterate and improve ML pipelines over it's lifetime using the insights derived from continuous stream of data points.  In LLM & RAGs, this can be applied by iterating and improving each components of LLM application from insights derived from production and feedback data.
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
