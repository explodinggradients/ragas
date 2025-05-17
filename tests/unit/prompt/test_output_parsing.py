import pytest
from pydantic import BaseModel, Field
import json

from ragas.prompt.utils import extract_json
from ragas.prompt.pydantic_prompt import RagasOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompt_values import StringPromptValue


class TestModel(BaseModel):
    """Test model for output parsing tests."""
    field1: str = Field(description="A string field")
    field2: int = Field(description="An integer field")


class TestExtractJson:
    """Test the extract_json function with various input formats."""

    def test_standard_json(self):
        """Test with standard JSON format."""
        text = '{"field1": "value1", "field2": 42}'
        result = extract_json(text)
        assert result == text
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["field1"] == "value1"
        assert parsed["field2"] == 42

    def test_json_in_markdown(self):
        """Test with JSON in markdown code block."""
        text = """
        Here's the JSON:
        ```json
        {"field1": "value1", "field2": 42}
        ```
        """
        result = extract_json(text)
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["field1"] == "value1"
        assert parsed["field2"] == 42

    def test_json_with_single_quotes(self):
        """Test with JSON using single quotes instead of double quotes."""
        text = "{'field1': 'value1', 'field2': 42}"
        result = extract_json(text)
        # Verify it's valid JSON after conversion
        parsed = json.loads(result)
        assert parsed["field1"] == "value1"
        assert parsed["field2"] == 42

    def test_json_with_trailing_comma(self):
        """Test with JSON containing trailing commas (invalid JSON but common in LLM outputs)."""
        text = '{"field1": "value1", "field2": 42,}'
        result = extract_json(text)
        # Verify it's valid JSON after fixing
        parsed = json.loads(result)
        assert parsed["field1"] == "value1"
        assert parsed["field2"] == 42

    def test_json_in_text(self):
        """Test with JSON embedded in text."""
        text = """
        The answer is as follows:
        {"field1": "value1", "field2": 42}
        Hope this helps!
        """
        result = extract_json(text)
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["field1"] == "value1"
        assert parsed["field2"] == 42


@pytest.mark.asyncio
async def test_ragas_output_parser_fallback(mocker):
    """Test that RagasOutputParser can handle malformed JSON with fallback mechanism."""
    # Create a parser
    parser = RagasOutputParser(pydantic_object=TestModel)
    
    # Mock the LLM to avoid actual calls
    mock_llm = mocker.MagicMock()
    mock_llm.generate = mocker.AsyncMock()
    
    # Test with malformed JSON that should trigger the fallback
    malformed_json = "This is not JSON at all"
    
    # Mock the fix_output_format_prompt.generate to return something that's still not valid
    mocker.patch(
        "ragas.prompt.pydantic_prompt.fix_output_format_prompt.generate",
        return_value=mocker.MagicMock(text="Still not valid JSON")
    )
    
    # The parser should use the fallback mechanism and return a valid model
    result = await parser.parse_output_string(
        output_string=malformed_json,
        prompt_value=StringPromptValue(text="test prompt"),
        llm=mock_llm,
        callbacks=None,
    )
    
    # Verify we got a valid model with default values
    assert isinstance(result, TestModel)
    assert result.field1 == "Unable to parse"
    assert result.field2 == 0