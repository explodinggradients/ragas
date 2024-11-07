
=== "OpenAI"
    Install the langchain-openai package

    ```bash
    pip install langchain-openai
    ```

    Ensure you have your OpenAI key ready and available in your environment.

    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "your-openai-key"
    ```
    Wrap the LLMs in `LangchainLLMWrapper` so that it can be used with ragas.

    ```python
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    ```


=== "Amazon Bedrock"
    Install the langchain-aws package

    ```bash
    pip install langchain-aws
    ```

    Then you have to set your AWS credentials and configurations

    ```python
    config = {
        "credentials_profile_name": "your-profile-name",  # E.g "default"
        "region_name": "your-region-name",  # E.g. "us-east-1"
        "llm": "your-llm-model-id",  # E.g "anthropic.claude-3-5-sonnet-20241022-v2:0"
        "embeddings": "your-embedding-model-id",  # E.g "amazon.titan-embed-text-v2:0"
        "temperature": 0.4,
    }
    ```

    Define your LLMs and wrap them in `LangchainLLMWrapper` so that it can be used with ragas.

    ```python
    from langchain_aws import ChatBedrockConverse
    from langchain_aws import BedrockEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    evaluator_llm = LangchainLLMWrapper(ChatBedrockConverse(
        credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        base_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
        model=config["llm"],
        temperature=config["temperature"],
    ))
    evaluator_embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
        credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        model_id=config["embeddings"],
    ))
    ```

    If you want more information on how to use other AWS services, please refer to the [langchain-aws](https://python.langchain.com/docs/integrations/providers/aws/) documentation.

=== "Azure OpenAI"
    Install the langchain-openai package

    ```bash
    pip install langchain-openai
    ```

    Ensure you have your Azure OpenAI key ready and available in your environment.

    ```python
    import os
    os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-key"

    # other configuration
    azure_config = {
        "base_url": "",  # your endpoint
        "model_deployment": "",  # your model deployment name
        "model_name": "",  # your model name
        "embedding_deployment": "",  # your embedding deployment name
        "embedding_name": "",  # your embedding name
    }

    ```

    Define your LLMs and wrap them in `LangchainLLMWrapper` so that it can be used with ragas.

    ```python
    from langchain_openai import AzureChatOpenAI
    from langchain_openai import AzureOpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["model_deployment"],
        model=azure_configs["model_name"],
        validate_base_url=False,
    ))

    # init the embeddings for answer_relevancy, answer_correctness and answer_similarity
    evaluator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["embedding_deployment"],
        model=azure_configs["embedding_name"],
    ))
    ```

    If you want more information on how to use other Azure services, please refer to the [langchain-azure](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/) documentation.


=== "Others"
    If you are using a different LLM provider and using Langchain to interact with it, you can wrap your LLM in `LangchainLLMWrapper` so that it can be used with ragas.

    ```python
    from ragas.llms import LangchainLLMWrapper
    evaluator_llm = LangchainLLMWrapper(your_llm_instance)
    ```

    For a more detailed guide, checkout [the guide on customizing models](../../howtos/customizations/customize_models/).

    If you using LlamaIndex, you can use the `LlamaIndexLLMWrapper` to wrap your LLM so that it can be used with ragas.

    ```python
    from ragas.llms import LlamaIndexLLMWrapper
    evaluator_llm = LlamaIndexLLMWrapper(your_llm_instance)
    ```

    For more information on how to use LlamaIndex, please refer to the [LlamaIndex Integration guide](../../howtos/integrations/_llamaindex/).

    If your still not able use Ragas with your favorite LLM provider, please let us know by by commenting on this [issue](https://github.com/explodinggradients/ragas/issues/1617) and we'll add support for it 🙂.