
# Explainability through Logging and tracing

Logging and tracing results from llm are important for any language model-based application. This is a tutorial on how to do tracing with Ragas. Ragas provides `callbacks` functionality which allows you to hook various tracers like Langmsith, wandb, etc easily.  In this notebook, I will be using Langmith for tracing

To set up Langsmith, we need to set some environment variables that it needs. For more information, you can refer to the [docs](https://docs.smith.langchain.com/)

```{code-block} bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

Now we have to import the required tracer from langchain, here we are using `LangChainTracer` but you can similarly use any tracer supported by langchain like [WandbTracer](https://python.langchain.com/docs/integrations/providers/wandb_tracing)

```{code-block} python
# langsmith
from langchain.callbacks.tracers import LangChainTracer

tracer = LangChainTracer(project_name="callback-experiments")
```

We now pass the tracer to the `callbacks` parameter when calling `evaluate`

```{code-block} python
from datasets import load_dataset
from ragas.metrics import context_precision
from ragas import evaluate

dataset = load_dataset("explodinggradients/amnesty_qa","english")
evaluate(dataset["train"],metrics=[context_precision],callbacks=[tracer])
```

```
{'context_precision': 1.0000}
```
![](./../../_static/imgs/trace-langsmith.png)
 

You can also write your own custom callbacks using langchain’s `BaseCallbackHandler`, refer [here](https://www.notion.so/Docs-logging-and-tracing-6f21cde9b3cb4d499526f48fd615585d?pvs=21) to read more about it.