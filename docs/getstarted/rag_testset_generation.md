## Testset Generation for RAG

This simple guide will help you generate a testset for evaluating your RAG pipeline using your own documents.

### Load Sample Documents

For the sake of this tutorial we will use sample documents from this [repository](https://huggingface.co/datasets/explodinggradients/Sample_Docs_Markdown). You can replace this with your own documents.

```bash
git clone https://huggingface.co/datasets/explodinggradients/Sample_Docs_Markdown
```


### Load documents
Now we will load the documents from the sample dataset using `DirectoryLoader`, which is one of document loaders from [langchain_community](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/). You may also use any loaders from [llama_index](https://docs.llamaindex.ai/en/stable/understanding/loading/llamahub/)

```python
from langchain_community.document_loaders import DirectoryLoader

path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()
```


### Choose your LLM

You may choose to use any [LLM of your choice]()

=== "OpenAI"
    This guide utilizes OpenAI for running some metrics, so ensure you have your OpenAI key ready and available in your environment.
    
    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "your-openai-key"
    ```
    Wrapp the LLMs in `LangchainLLMWrapper`
    ```python
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    ```




=== "AWS Bedrock"
    First you have to set your AWS credentials and configurations
    
    ```python
    config = {
        "credentials_profile_name": "your-profile-name",  # E.g "default"
        "region_name": "your-region-name",  # E.g. "us-east-1"
        "model_id": "your-model-id",  # E.g "anthropic.claude-v2"
        "model_kwargs": {"temperature": 0.4},
    }
    ```
    define you LLMs
    ```python
    from langchain_aws.chat_models import BedrockChat
    from ragas.llms import LangchainLLMWrapper
    generator_llm = LangchainLLMWrapper(BedrockChat(
        credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        endpoint_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
        model_id=config["model_id"],
        model_kwargs=config["model_kwargs"],
    ))
    ```


### Generate Testset

Now we will run the test generation using the loaded documents and the LLM setup. If you have used `llama_index` to load documents, please use `generate_with_llama_index_docs` method instead.

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm)
dataset = generator.generate_with_langchain_docs(docs, test_size=10)
```

### Export

You may now export and inspect the generated testset.
    
```python
dataset.to_pandas()
```