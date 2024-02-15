from collections import namedtuple

import pytest
from datasets import Dataset

from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.validation import (
    remap_column_names,
    validate_column_dtypes,
    validate_evaluation_modes,
)

CaseToTest = namedtuple(
    "TestCase", ["q", "a", "c", "g", "is_valid_columns", "metrics", "is_valid_metrics"]
)

TEST_CASES = [
    CaseToTest("a", "b", ["c"], None, True, [faithfulness], True),
    CaseToTest("a", "b", ["c"], "g", True, [faithfulness], True),
    CaseToTest("a", None, ["c"], "g", True, [context_precision], True),
    CaseToTest("a", "b", "c", "g", False, [context_precision], True),
    CaseToTest(
        "a", None, [["c"]], None, False, [context_precision, answer_relevancy], False
    ),
    CaseToTest("a", None, ["c"], ["g"], False, [context_precision], True),
    CaseToTest("a", None, ["c"], [["g"]], False, [context_precision], True),
    CaseToTest(1, None, ["c"], "g", False, [context_precision], True),
    CaseToTest(1, None, None, None, False, [context_precision], False),
]


@pytest.mark.parametrize("testcase", TEST_CASES)
def test_validate_column_dtypes(testcase):
    dataset_dict = {}
    if testcase.q is not None:
        dataset_dict["question"] = [testcase.q]
    if testcase.a is not None:
        dataset_dict["answer"] = [testcase.a]
    if testcase.c is not None:
        dataset_dict["contexts"] = [testcase.c]
    if testcase.g is not None:
        dataset_dict["ground_truth"] = [testcase.g]

    test_dataset = Dataset.from_dict(dataset_dict)
    if testcase.is_valid_columns:
        validate_column_dtypes(test_dataset)
    else:
        with pytest.raises(ValueError):
            validate_column_dtypes(test_dataset)


@pytest.mark.parametrize("testcase", TEST_CASES)
def test_validate_columns_and_metrics(testcase):
    dataset_dict = {}
    if testcase.q is not None:
        dataset_dict["question"] = [testcase.q]
    if testcase.a is not None:
        dataset_dict["answer"] = [testcase.a]
    if testcase.c is not None:
        dataset_dict["contexts"] = [testcase.c]
    if testcase.g is not None:
        dataset_dict["ground_truth"] = [testcase.g]
    test_dataset = Dataset.from_dict(dataset_dict)

    if testcase.is_valid_metrics:
        validate_evaluation_modes(test_dataset, testcase.metrics)
    else:
        with pytest.raises(ValueError):
            validate_evaluation_modes(test_dataset, testcase.metrics)


column_maps = [
    {
        "question": "query",
        "answer": "rag_answer",
        "contexts": "rag_contexts",
        "ground_truth": "original_answer",
    },  # all columns present
    {
        "question": "query",
        "answer": "rag_answer",
    },  # subset of columns present
]


@pytest.mark.parametrize("column_map", column_maps)
def test_column_remap(column_map):
    """
    test cases:
    - extra columns present in the dataset
    - not all columsn selected
    - column names are different
    """
    TEST_DATASET = Dataset.from_dict(
        {
            "query": [""],
            "rag_answer": [""],
            "rag_contexts": [[""]],
            "original_answer": [""],
            "another_column": [""],
            "rag_answer_v2": [""],
            "rag_contexts_v2": [[""]],
        }
    )
    remapped_dataset = remap_column_names(TEST_DATASET, column_map)
    assert all(col in remapped_dataset.column_names for col in column_map.keys())


def test_column_remap_omit():
    TEST_DATASET = Dataset.from_dict(
        {
            "query": [""],
            "answer": [""],
            "contexts": [[""]],
        }
    )

    column_map = {
        "question": "query",
        "contexts": "contexts",
        "answer": "answer",
    }

    remapped_dataset = remap_column_names(TEST_DATASET, column_map)
    assert remapped_dataset.column_names == ["question", "answer", "contexts"]
