(get-started-monitoring)=
# Monitoring

Maintaining the quality and performance of an LLM application in a production environment can be challenging. Ragas provides with basic building blocks that you can use for production quality monitoring, offering valuable insights into your application's performance. This is achieved by constructing custom, smaller, more cost-effective, and faster models.

:::{note}
This is feature is still in beta access. You can requests for 
[**early access**](https://calendly.com/shahules/30min) to get access
:::

The Ragas metrics can also be used with other LLM observability tools like
[Langsmith](https://www.langchain.com/langsmith) and
[Langfuse](https://langfuse.com/) to get model-based feedback about various
aspects of you application like those mentioned below

## What can be monitored

```{admonition} **Faithfulness**
:class: note

This feature assists in identifying and quantifying instances of hallucinations.
```

```{admonition} **Bad retrieval**
:class: note

This feature helps identify and quantify poor context retrievals.
```

```{admonition} **Bad response**
:class: note

This feature helps in recognizing and quantifying evasive, harmful, or toxic responses.
```

```{admonition} **Bad format**
:class: note

This feature helps in detecting and quantifying responses with incorrect formatting.
```

```{admonition} **Custom use-case**
:class: hint

For monitoring other critical aspects that are specific to your use case. [Talk to founders](https://calendly.com/shahules/30min)
