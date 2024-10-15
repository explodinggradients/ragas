
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
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    ```


=== "AWS Bedrock"
    First you have to set your AWS credentials and configurations

    ```python
    config = {
        "credentials_profile_name": "your-profile-name",  # E.g "default"
        "region_name": "your-region-name",  # E.g. "us-east-1"
        "llm": "your-llm-model-id",  # E.g "anthropic.claude-3-5-sonnet-20240620-v1:0"
        "embeddings": "your-embedding-model-id",  # E.g "amazon.titan-embed-text-v2:0"
        "temperature": 0.4,
    }
    ```
    define you LLMs
    ```python
    from langchain_aws import ChatBedrockConverse
    from ragas.llms import LangchainLLMWrapper
    evaluator_llm = LangchainLLMWrapper(ChatBedrockConverse(
        credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        base_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
        model=config["llm"],
        temperature=config["temperature"],
    ))
    ```