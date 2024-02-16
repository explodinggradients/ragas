(get-started-monitoring)=
# Monitor your RAG in Production

Maintaining the quality and performance of a RAG application in a production environment can be challenging. RAG currently provides the essential building blocks that you can use for production-quality monitoring, offering valuable insights into your application's performance. However, we are also working towards building a more advanced production monitoring solution by addressing three questions:

1. How can we ensure the distribution of your production dataset remains consistent with your test set?
2. How can we effectively extract insights from explicit and implicit signals your users provide to infer the quality of your RAG application and identify areas that require attention?
3. How can we construct custom, smaller, more cost-effective and faster models for evaluation and more advanced test set generation?

:::{note}
We are still developing and gathering feedback for upcoming releases. You can request 
[**early access**](https://calendly.com/shahules/30min) to try it out or share the challenges you face in this area. We would love to hear your thoughts and challenges.
:::

Additionally, you can use the RAG metrics with other Machine Learning Model (MLM) observability tools like
- [Langsmith](../howtos/integrations/langsmith.ipynb)
- [Phoenix (Arize)](https://github.com/Arize-ai/phoenix)
- [Langfuse](../howtos/integrations/langfuse.ipynb)
- [OpenLayer](https://openlayer.com/)

to get model-based feedback about various aspects of your application, such as those mentioned below:

## Aspects to Monitor

1. Faithfulness: This feature assists in identifying and quantifying instances of hallucinations.
2. Bad retrieval: This feature helps identify and quantify poor context retrievals.
3. Bad response: This feature assists in recognizing and quantifying evasive, harmful, or toxic responses.
4. Bad format: This feature enables the detection and quantification of responses with incorrect formatting.
5. Custom use-case: For monitoring other critical aspects that are specific to your use case, [Talk to founders](https://calendly.com/shahules/30min).
