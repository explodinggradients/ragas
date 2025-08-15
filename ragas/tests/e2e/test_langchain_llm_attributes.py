import pytest

try:
    from langchain_anthropic import ChatAnthropic  # type: ignore
    from langchain_aws import ChatBedrock, ChatBedrockConverse  # type: ignore
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_google_vertexai import ChatVertexAI  # type: ignore
    from langchain_openai import ChatOpenAI  # type: ignore

    LANGCHAIN_AVAILABLE = True

    models = [
        ChatOpenAI(model="gpt-4o"),
        # AzureChatOpenAI(model="gpt-4o", api_version="2024-04-09"),
        ChatGoogleGenerativeAI(model="gemini-1.5-pro"),
        ChatAnthropic(
            model_name="claude-3-5-sonnet-20240620",
            timeout=10,
            stop=["\n\n"],
            temperature=0.5,
        ),
        ChatBedrock(model="anthropic.claude-3-5-sonnet-20240620"),
        ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20240620"),
        ChatVertexAI(model="gemini-1.5-pro"),
    ]

except ImportError:
    LANGCHAIN_AVAILABLE = False
    models = []

    # Skip all tests if langchain not available
    pytestmark = pytest.mark.skip("langchain dependencies not available")


@pytest.mark.parametrize("model", models)
def test_langchain_chat_models_have_temperature(model):
    assert hasattr(model, "temperature")
    model.temperature = 0.5
    assert model.temperature == 0.5


@pytest.mark.parametrize("model", models)
def test_langchain_chat_models_have_n(model):
    assert hasattr(model, "n")
    model.n = 2
    assert model.n == 2
