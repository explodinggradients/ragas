import pytest
import json
from ragas.prompt.utils import extract_json
from ragas.run_config import RunConfig


def test_extract_json_with_markdown():
    """Test extracting JSON from markdown code blocks."""
    text = """
    Here's the JSON:
    ```json
    {"key": "value", "array": [1, 2, 3]}
    ```
    """
    result = extract_json(text)
    assert json.loads(result) == {"key": "value", "array": [1, 2, 3]}


def test_extract_json_with_single_quotes():
    """Test extracting JSON with single quotes (common in local model outputs)."""
    text = "{'key': 'value', 'array': [1, 2, 3]}"
    result = extract_json(text)
    assert json.loads(result) == {"key": "value", "array": [1, 2, 3]}


def test_extract_json_with_trailing_commas():
    """Test extracting JSON with trailing commas (common in local model outputs)."""
    text = '{"key": "value", "array": [1, 2, 3,],}'
    result = extract_json(text)
    assert json.loads(result) == {"key": "value", "array": [1, 2, 3]}


def test_run_config_local_model():
    """Test that local model flag increases timeout."""
    # Default config
    config = RunConfig()
    assert config.timeout == 180
    
    # Local model config
    local_config = RunConfig(is_local_model=True)
    assert local_config.timeout == 600  # Should be increased to 10 minutes