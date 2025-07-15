# Prompt Evaluation

At the end of this tutorial you’ll learn how to iterate on a single prompt using evaluation driven development. 

```mermaid
flowchart LR
    A["'This movie was amazing!<br/>Great acting and plot.'"] --> B["Classifier Prompt"]
    B --> C["Positive"]
```


Setup your OpenAI API key

```bash
export OPENAI_API_KEY = "your_openai_api_key"
```


