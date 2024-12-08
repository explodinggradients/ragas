## Customize Models

Ragas may use a LLM and or Embedding for evaluation and synthetic data generation. Both of these models can be customised according to you availabiity. 

!!! note
    Ragas supports all the [LLMs](https://python.langchain.com/docs/integrations/chat/) and [Embeddings](https://python.langchain.com/docs/integrations/text_embedding/) available in langchain

- `BaseRagasLLM` and `BaseRagasEmbeddings` are the base classes Ragas uses internally for LLMs and Embeddings. Any custom LLM or Embeddings should be a subclass of these base classes.  

- If you are using Langchain, you can pass the Langchain LLM and Embeddings directly and Ragas will wrap it with `LangchainLLMWrapper` or `LangchainEmbeddingsWrapper` as needed.

## Examples

- [Azure OpenAI](#azure-openai)
- [Google Vertex](#google-vertex)
- [AWS Bedrock](#aws-bedrock)


### Azure OpenAI

```bash
pip install langchain_openai
```

```python

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

azure_configs = {
    "base_url": "https://<your-endpoint>.openai.azure.com/",
    "model_deployment": "your-deployment-name",
    "model_name": "your-model-name",
    "embedding_deployment": "your-deployment-name",
    "embedding_name": "text-embedding-ada-002",  # most likely
}


azure_llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["model_deployment"],
    model=azure_configs["model_name"],
    validate_base_url=False,
)

# init the embeddings for answer_relevancy, answer_correctness and answer_similarity
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["embedding_deployment"],
    model=azure_configs["embedding_name"],
)

azure_llm = LangchainLLMWrapper(azure_llm)
azure_embeddings = LangchainEmbeddingsWrapper(azure_embeddings)
```
Yay! Now are you ready to use ragas with Azure OpenAI endpoints

### Google Vertex

```bash
!pip install langchain_google_vertexai
```

```python
import google.auth
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.outputs import LLMResult, ChatGeneration

config = {
    "project_id": "<your-project-id>",
    "chat_model_id": "gemini-1.5-pro-002",
    "embedding_model_id": "text-embedding-005",
}

# authenticate to GCP
creds, _ = google.auth.default(quota_project_id=config["project_id"])

# create Langchain LLM and Embeddings
vertextai_llm = ChatVertexAI(
    credentials=creds,
    model_name=config["chat_model_id"],
)
vertextai_embeddings = VertexAIEmbeddings(
    credentials=creds, model_name=config["embedding_model_id"]
)

# Create a custom is_finished_parser to capture Gemini generation completion signals
def gemini_is_finished_parser(response: LLMResult) -> bool:
    is_finished_list = []
    for g in response.flatten():
        resp = g.generations[0][0]
        
        # Check generation_info first
        if resp.generation_info is not None:
            finish_reason = resp.generation_info.get("finish_reason")
            if finish_reason is not None:
                is_finished_list.append(
                    finish_reason in ["STOP", "MAX_TOKENS"]
                )
                continue
                
        # Check response_metadata as fallback
        if isinstance(resp, ChatGeneration) and resp.message is not None:
            metadata = resp.message.response_metadata
            if metadata.get("finish_reason"):
                is_finished_list.append(
                    metadata["finish_reason"] in ["STOP", "MAX_TOKENS"]
                )
            elif metadata.get("stop_reason"):
                is_finished_list.append(
                    metadata["stop_reason"] in ["STOP", "MAX_TOKENS"] 
                )
        
        # If no finish reason found, default to True
        if not is_finished_list:
            is_finished_list.append(True)
            
    return all(is_finished_list)


vertextai_llm = LangchainLLMWrapper(vertextai_llm, is_finished_parser=gemini_is_finished_parser)
vertextai_embeddings = LangchainEmbeddingsWrapper(vertextai_embeddings)
```
Yay! Now are you ready to use ragas with Google VertexAI endpoints

### AWS Bedrock

```bash
pip install langchain_aws
```

```python
from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

config = {
    "credentials_profile_name": "your-profile-name",  # E.g "default"
    "region_name": "your-region-name",  # E.g. "us-east-1"
    "llm": "your-llm-model-id",  # E.g "anthropic.claude-3-5-sonnet-20241022-v2:0"
    "embeddings": "your-embedding-model-id",  # E.g "amazon.titan-embed-text-v2:0"
    "temperature": 0.4,
}

bedrock_llm = ChatBedrockConverse(
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
    base_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
    model=config["llm"],
    temperature=config["temperature"],
)

# init the embeddings
bedrock_embeddings = BedrockEmbeddings(
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
    model_id=config["embeddings"],
)

bedrock_llm = LangchainLLMWrapper(bedrock_llm)
bedrock_embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)
```
Yay! Now are you ready to use ragas with AWS Bedrock endpoints
