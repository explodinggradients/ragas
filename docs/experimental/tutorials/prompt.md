# Prompt Evaluation

At the end of this tutorial you’ll learn how to iterate on a single prompt using evaluation driven development. 

```mermaid
flowchart LR
    A["Input:<br/>Movie Review Text<br/><br/>'This movie was amazing!<br/>Great acting and plot.'"] --> B["Movie Review<br/>Classifier Prompt<br/><br/>Analyze sentiment...<br/>Positive/Negative"]
    B --> C["Output:<br/>Classification Result<br/><br/>Positive"]
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e8f5e8
```


Setup your OpenAI API key

```bash
export OPENAI_API_KEY = "your_openai_api_key"
```


