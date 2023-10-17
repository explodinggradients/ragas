(ragas-metrics)=
# Metrics

## Component-Wise Evaluation

Just like in any machine learning system, the performance of individual components within the LLM and RAG pipeline has a significant impact on the overall experience. Ragas offers metrics tailored for evaluating each component of your RAG pipeline in isolation.

<p align="center">
<img src="../../_static/imgs/component-wise-metrics.png" alt="evol-generate" width="600" height="400" />
</p>

- [Faithfulnesss](faithfulness.md)
- [Answer relevancy](answer_relevance.md)
- [Context recall](context_recall.md)
- [Context precision](context_precision.md)

## End-to-End Evaluation

Evaluating the end-to-end performance of a pipeline is also crucial, as it directly affects the user experience. Ragas provides metrics that can be employed to assess the overall performance of your pipeline, ensuring a comprehensive evaluation.

- [Answer semantic similarity](semantic_similarity.md)
- [Answer correctness](answer_correctness.md)

```{toctree}
:maxdepth: 1
:hidden:

faithfulness
answer_relevance
context_precision
context_recall
semantic_similarity
answer_correctness
critique

```
