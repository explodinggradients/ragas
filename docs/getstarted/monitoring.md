(get-started-monitoring)=
# Monitor Your RAG in Production

Maintaining the quality and performance of a RAG application in a production environment is challenging. RAG currently provides the essential building blocks for production-quality monitoring, offering valuable insights into your application's performance. However, we are also working towards building a more advanced production monitoring solution by addressing three key areas:

1. How to ensure the distribution of your production dataset remains consistent with your test set.
2. How to effectively extract insights from the explicit and implicit signals your users provide to infer the quality of your RAG application and identify areas that require attention.
3. How to construct custom, smaller, more cost-effective, and faster models for evaluation and advanced test set generation.

:::{note}
We are still developing and gathering feedback for upcoming releases. You can request 
[**early access**](https://cal.com/shahul-ragas/30min) to try it out or share the challenges you face in this area. We would love to hear your thoughts and challenges.
:::

In addition, you can use the RAG metrics with other LLM observability tools like:

- [Langsmith](../howtos/integrations/langsmith.ipynb)
- [Phoenix (Arize)](../howtos/integrations/ragas-arize.ipynb)
- [Langfuse](../howtos/integrations/langfuse.ipynb)
- [OpenLayer](https://openlayer.com/)

These tools can provide model-based feedback about various aspects of your application, such as the ones mentioned below:

## Aspects to Monitor

1. Faithfulness: This feature assists in identifying and quantifying instances of hallucination.
2. Bad Retrieval: This feature helps identify and quantify poor context retrievals.
3. Bad Response: This feature assists in recognizing and quantifying evasive, harmful, or toxic responses.
4. Bad Format: This feature enables the detection and quantification of responses with incorrect formatting.
5. Custom Use-Case: For monitoring other critical aspects that are specific to your use-case, [Talk to the founders](https://cal.com/shahul-ragas/30min).
