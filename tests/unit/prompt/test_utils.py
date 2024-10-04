from collections import namedtuple

import pytest
from pydantic import BaseModel

from ragas.prompt.utils import get_all_strings, update_strings


class Category(BaseModel):
    category: str
    name: str = "good_name"
    is_good: bool = True
    number: int = 1


class Categories(BaseModel):
    list_of_categories: list[Category]
    list_of_names: list[str] = ["good_name1", "good_name2", "good_name3"]


old_strings = ["old1", "old2", "old3"]
new_strings = ["new1", "new2", "new3"]

TestCase = namedtuple("TestCase", ["obj", "old_strings", "new_strings"])

test_cases = [
    TestCase(
        obj={
            "a": "old 1",
            "b": "old 2",
            "c": ["old 1", "old 2", "old 3"],
            "d": {"e": "old 2"},
        },
        old_strings=["old 1", "old 2", "old 1", "old 2", "old 3", "old 2"],
        new_strings=["old_1", "old_2", "old_1", "old_2", "old_3", "old_2"],
    ),
    TestCase(
        obj=Categories(
            list_of_categories=[
                Category(category="old1", name="name_old1"),
                Category(category="old2", name="name_old2"),
                Category(category="old3", name="name_old3"),
                Category(category="old1", name="name_old1"),
            ]
        ),
        old_strings=["old 1", "old 2", "old 3", "old 1"],
        new_strings=["old_1", "old_2", "old_3", "old_1"],
    ),
    TestCase(
        obj=[
            Category(category="old 1", is_good=True, number=1),
            Category(category="old 2", is_good=True, number=2),
            Category(category="old 3", is_good=True, number=3),
            Category(category="old 1", is_good=True, number=4),
        ],
        old_strings=["old 1", "old 2", "old 3", "old 1"],
        new_strings=["old_1", "old_2", "old_3", "old_1"],
    ),
]


@pytest.mark.parametrize(
    "obj, expected",
    [
        (
            {
                "a": "old1",
                "b": "old2",
                "c": ["old1", "old2", "old3"],
                "d": {"e": "old2"},
            },
            ["old1", "old2", "old1", "old2", "old3", "old2"],
        ),
        (
            [
                Category(category="old1", is_good=True, number=1),
                Category(category="old2", is_good=True, number=2),
                Category(category="old3", is_good=True, number=3),
                Category(category="old1", is_good=True, number=4),
            ],
            [
                "old1",
                "good_name",
                "old2",
                "good_name",
                "old3",
                "good_name",
                "old1",
                "good_name",
            ],
        ),
        (
            Categories(
                list_of_categories=[
                    Category(category="old1", name="name_old1"),
                    Category(category="old2", name="name_old2"),
                    Category(category="old3", name="name_old3"),
                    Category(category="old1", name="name_old1"),
                ]
            ),
            [
                "old1",
                "name_old1",
                "old2",
                "name_old2",
                "old3",
                "name_old3",
                "old1",
                "name_old1",
                "good_name1",
                "good_name2",
                "good_name3",
            ],
        ),
    ],
)
def test_get_all_strings(obj, expected):
    assert get_all_strings(obj) == expected


def test_update_strings():
    obj = {"a": "old1", "b": "old2", "c": ["old1", "old2", "old3"], "d": {"e": "old2"}}
    pass
