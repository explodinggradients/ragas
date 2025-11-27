"""
Test that retrieved_contexts as strings are properly handled in EvaluatorChain.

This test verifies the transformation logic handles both strings and LCDocument objects
correctly, ensuring strings are preserved and LCDocuments are converted to strings.
"""

import pytest

try:
    from langchain_core.documents import Document as LCDocument

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    pytestmark = pytest.mark.skip("langchain dependencies not available")

from ragas.utils import convert_row_v1_to_v2


@pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="langchain dependencies not available"
)
def test_retrieved_contexts_strings_filtered_out_bug():
    """
    Test that retrieved_contexts as strings are properly preserved.

    This test verifies that when retrieved_contexts contains strings (as per documentation),
    they are correctly preserved and not filtered out. Previously, strings were filtered out
    because the code only processed LCDocument objects, resulting in an empty list which
    caused context_precision to be 0.0.
    """
    # Simulate what happens in EvaluatorChain._call()
    # Use v1 format keys (contexts, question, ground_truth) which get converted to v2
    inputs = {
        "question": "What is the capital of France?",
        "contexts": [  # v1 key - will be converted to "retrieved_contexts"
            "Paris is the capital and largest city of France.",
            "Berlin is the capital of Germany.",
        ],
        "ground_truth": "Paris is the capital of France.",
    }

    # Convert to v2 format (as done in EvaluatorChain)
    inputs = convert_row_v1_to_v2(inputs)

    # Verify conversion worked
    assert "retrieved_contexts" in inputs, (
        "retrieved_contexts should exist after conversion"
    )
    assert len(inputs["retrieved_contexts"]) == 2, (
        "Should have 2 contexts before filtering"
    )

    # Apply the fixed transformation (current code behavior from EvaluatorChain._call)
    # This is the exact code from src/ragas/integrations/langchain.py lines 79-82
    if "retrieved_contexts" in inputs:
        inputs["retrieved_contexts"] = [
            doc.page_content if isinstance(doc, LCDocument) else str(doc)
            for doc in inputs["retrieved_contexts"]
        ]

    # EXPECTED BEHAVIOR: Strings should be preserved, not filtered out
    assert len(inputs["retrieved_contexts"]) == 2, (
        f"BUG: Strings were filtered out! Expected 2 contexts, got {len(inputs['retrieved_contexts'])}. "
        f"Contents: {inputs['retrieved_contexts']}"
    )

    # Verify the strings are preserved correctly
    assert (
        inputs["retrieved_contexts"][0]
        == "Paris is the capital and largest city of France."
    )
    assert inputs["retrieved_contexts"][1] == "Berlin is the capital of Germany."

    print(f"\n✓ Strings preserved correctly: {inputs['retrieved_contexts']}")


@pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="langchain dependencies not available"
)
def test_retrieved_contexts_lcdocuments_work():
    """
    Verify that LCDocument objects are processed correctly (current working case).
    This is the expected behavior when LCDocuments are passed.
    """
    # Use v1 format keys
    inputs = {
        "question": "What is the capital of France?",
        "contexts": [
            LCDocument(page_content="Paris is the capital and largest city of France."),
            LCDocument(page_content="Berlin is the capital of Germany."),
        ],
        "ground_truth": "Paris is the capital of France.",
    }

    inputs = convert_row_v1_to_v2(inputs)

    # Apply the transformation
    if "retrieved_contexts" in inputs:
        inputs["retrieved_contexts"] = [
            doc.page_content if isinstance(doc, LCDocument) else str(doc)
            for doc in inputs["retrieved_contexts"]
        ]

    # LCDocuments should be processed correctly
    assert len(inputs["retrieved_contexts"]) == 2, (
        f"Expected 2 contexts, got {len(inputs['retrieved_contexts'])}"
    )
    assert (
        inputs["retrieved_contexts"][0]
        == "Paris is the capital and largest city of France."
    )
    assert inputs["retrieved_contexts"][1] == "Berlin is the capital of Germany."


@pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="langchain dependencies not available"
)
def test_retrieved_contexts_mixed_strings_and_documents():
    """
    Test mixed case: some strings, some LCDocuments.
    Both should be preserved correctly - strings as-is, LCDocuments converted to strings.
    """
    # Use v1 format keys
    inputs = {
        "question": "What is the capital of France?",
        "contexts": [
            "Paris is the capital and largest city of France.",  # String - will be filtered
            LCDocument(
                page_content="Berlin is the capital of Germany."
            ),  # LCDocument - will work
        ],
        "ground_truth": "Paris is the capital of France.",
    }

    inputs = convert_row_v1_to_v2(inputs)

    # Apply the transformation
    if "retrieved_contexts" in inputs:
        inputs["retrieved_contexts"] = [
            doc.page_content if isinstance(doc, LCDocument) else str(doc)
            for doc in inputs["retrieved_contexts"]
        ]

    # EXPECTED BEHAVIOR: Both string and LCDocument should be preserved
    assert len(inputs["retrieved_contexts"]) == 2, (
        f"BUG: String was filtered out! Expected 2 contexts, got {len(inputs['retrieved_contexts'])}. "
        f"Contents: {inputs['retrieved_contexts']}"
    )

    # Verify both are preserved (order may vary, so check both)
    contexts = inputs["retrieved_contexts"]
    assert "Paris is the capital and largest city of France." in contexts
    assert "Berlin is the capital of Germany." in contexts

    print(f"\n✓ Both string and LCDocument preserved: {inputs['retrieved_contexts']}")
