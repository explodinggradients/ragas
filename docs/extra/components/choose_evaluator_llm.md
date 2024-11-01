
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

    then you have to set your AWS credentials and configurations

    ```python
    config = {
        "credentials_profile_name": "your-profile-name",  # E.g "default"
        "region_name": "your-region-name",  # E.g. "us-east-1"
        "llm": "your-llm-model-id",  # E.g "anthropic.claude-3-5-sonnet-20241022-v2:0"
        "embeddings": "your-embedding-model-id",  # E.g "amazon.titan-embed-text-v2:0"
        "temperature": 0.4,
    }
    ```

    define you LLMs and wrap them in `LangchainLLMWrapper` so that it can be used with ragas.

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