from collections import namedtuple

import pytest
from pydantic import BaseModel

from ragas.prompt.utils import get_all_strings, update_strings


class Category(BaseModel):
    category: str
    name: str = "good name"
    is_good: bool = True
    number: int = 1


class Categories(BaseModel):
    list_of_categories: list[Category]
    list_of_names: list[str] = ["good_name1", "good_name2", "good_name3"]


old_strings = ["old1", "old2", "old3"]
new_strings = ["new1", "new2", "new3"]

OurTestCase = namedtuple("OurTestCase", ["obj", "old_strings", "new_strings"])

test_cases = [
    OurTestCase(
        obj={
            "a": "old 1",
            "b": "old 2",
            "c": ["old 1", "old 2", "old 3"],
            "d": {"e": "old 2"},
        },
        old_strings=["old 1", "old 2", "old 1", "old 2", "old 3", "old 2"],
        new_strings=["old_1", "old_2", "old_1", "old_2", "old_3", "old_2"],
    ),
    OurTestCase(
        obj=Categories(
            list_of_categories=[
                Category(category="old 1", name="name old1"),
                Category(category="old 2", name="name old2"),
                Category(category="old 3", name="name old3"),
                Category(category="old 1", name="name old1"),
            ],
            list_of_names=["name 1", "name 2", "name 3"],
        ),
        old_strings=[
            "old 1",
            "name old1",
            "old 2",
            "name old2",
            "old 3",
            "name old3",
            "old 1",
            "name old1",
            "name 1",
            "name 2",
            "name 3",
        ],
        new_strings=[
            "old_1",
            "name old1",
            "old_2",
            "name old2",
            "old_3",
            "name old3",
            "old_1",
            "name old1",
            "name 1",
            "name 2",
            "name 3",
        ],
    ),
    OurTestCase(
        obj=[
            Category(category="old 1", is_good=True, number=1),
            Category(category="old 2", is_good=True, number=2),
            Category(category="old 3", is_good=True, number=3),
            Category(category="old 1", is_good=True, number=4),
        ],
        old_strings=[
            "old 1",
            "good name",
            "old 2",
            "good name",
            "old 3",
            "good name",
            "old 1",
            "good name",
        ],
        new_strings=[
            "old_1",
            "good_name",
            "old_2",
            "good_name",
            "old_3",
            "good_name",
            "old_1",
            "good_name",
        ],
    ),
]


@pytest.mark.parametrize(
    "obj, expected",
    [(test_case.obj, test_case.old_strings) for test_case in test_cases],
)
def test_get_all_strings(obj, expected):
    assert get_all_strings(obj) == expected


@pytest.mark.parametrize(
    "obj, old_strings, new_strings",
    [
        (test_case.obj, test_case.old_strings, test_case.new_strings)
        for test_case in test_cases
    ],
)
def test_update_strings(obj, old_strings, new_strings):
    updated_obj = update_strings(obj, old_strings, new_strings)

    assert get_all_strings(updated_obj) == new_strings
    assert get_all_strings(obj) == old_strings
