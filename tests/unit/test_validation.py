import pytest
from datasets import Dataset

from ragas.validation import validate_column_dtypes

test_dataset = Dataset.from_dict(
    {
        "question": ["a"],
        "contexts": [["a"]],
    }
)

TEST_CASES = [
    ("a", "b", ["c"], None, True),
    ("a", "b", ["c"], ["g"], True),
    ("a", None, ["c"], None, True),
    ("a", None, "c", None, False),
    ("a", None, [["c"]], None, False),
    ("a", None, ["c"], "g", False),
    ("a", None, ["c"], [["g"]], False),
    (1, None, ["c"], ["g"], False),
    (1, None, None, None, False),
]


@pytest.mark.parametrize("q,a,c,g,is_valid", TEST_CASES)
def test_validate_column_dtypes(q, a, c, g, is_valid):
    dataset_dict = {}
    if q is not None:
        dataset_dict["question"] = [q]
    if a is not None:
        dataset_dict["answer"] = [a]
    if c is not None:
        dataset_dict["contexts"] = [c]
    if g is not None:
        dataset_dict["ground_truths"] = [g]

    test_dataset = Dataset.from_dict(dataset_dict)
    if is_valid:
        validate_column_dtypes(test_dataset)
    else:
        with pytest.raises(ValueError):
            validate_column_dtypes(test_dataset)
