from collections import namedtuple

import pytest
from datasets import Dataset

from ragas.metrics import answer_relevancy, context_relevancy, faithfulness
from ragas.validation import validate_column_dtypes, validate_evaluation_modes

CaseToTest = namedtuple(
    "TestCase", ["q", "a", "c", "g", "is_valid_columns", "metrics", "is_valid_metrics"]
)

TEST_CASES = [
    CaseToTest("a", "b", ["c"], None, True, [faithfulness], True),
    CaseToTest("a", "b", ["c"], ["g"], True, [faithfulness], True),
    CaseToTest("a", None, ["c"], None, True, [context_relevancy], True),
    CaseToTest("a", None, "c", None, False, [context_relevancy], True),
    CaseToTest(
        "a", None, [["c"]], None, False, [context_relevancy, answer_relevancy], False
    ),
    CaseToTest("a", None, ["c"], "g", False, [context_relevancy], True),
    CaseToTest("a", None, ["c"], [["g"]], False, [context_relevancy], True),
    CaseToTest(1, None, ["c"], ["g"], False, [context_relevancy], True),
    CaseToTest(1, None, None, None, False, [context_relevancy], False),
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
        dataset_dict["ground_truths"] = [testcase.g]

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
        dataset_dict["ground_truths"] = [testcase.g]
    test_dataset = Dataset.from_dict(dataset_dict)

    if testcase.is_valid_metrics:
        validate_evaluation_modes(test_dataset, testcase.metrics)
    else:
        with pytest.raises(ValueError):
            validate_evaluation_modes(test_dataset, testcase.metrics)
