
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
        "model_id": "your-model-id",  # E.g "anthropic.claude-v2"
        "model_kwargs": {"temperature": 0.4},
    }
    ```
    define you LLMs
    ```python
    from langchain_aws.chat_models import BedrockChat
    from ragas.llms import LangchainLLMWrapper
    evaluator_llm = LangchainLLMWrapper(BedrockChat(
        credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        endpoint_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
        model_id=config["model_id"],
        model_kwargs=config["model_kwargs"],
    ))
    ```