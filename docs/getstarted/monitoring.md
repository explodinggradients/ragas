(get-started-monitoring)=
# Monitoring

Maintaining the quality and performance of an LLM application in a production environment can be challenging. Ragas provides with basic building blocks that you can use for production quality monitoring, offering valuable insights into your application's performance. This is achieved by constructing custom, smaller, more cost-effective, and faster models.

:::{note}
This is feature is still in beta access. You can requests for 
[**early access**](https://calendly.com/shahules/30min) to try it out.
:::

The Ragas metrics can also be used with other LLM observability tools like
[Langsmith](https://www.langchain.com/langsmith) and
[Langfuse](https://langfuse.com/) to get model-based feedback about various
aspects of you application like those mentioned below

:::{seealso}
[Langfuse Integration](../howtos/integrations/langfuse.ipynb) to see Ragas
monitoring in action within the Langfuse dashboard and how to set it up
:::

## Aspects to Monitor

1. Faithfulness: This feature assists in identifying and quantifying instances of hallucinations.
2. Bad retrieval: This feature helps identify and quantify poor context retrievals.
3. Bad response: This feature helps in recognizing and quantifying evasive, harmful, or toxic responses.
4. Bad format: This feature helps in detecting and quantifying responses with incorrect formatting.
5. Custom use-case: For monitoring other critical aspects that are specific to your use case. [Talk to founders](https://calendly.com/shahules/30min)
