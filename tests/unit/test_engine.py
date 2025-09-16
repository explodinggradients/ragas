import asyncio
import types
import typing as t

import pytest

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms.base import BaseGraphTransformation
from ragas.testset.transforms.engine import Parallel, apply_transforms, get_desc


class DummyTransformation(BaseGraphTransformation):
    def __init__(self, name="Dummy"):
        self.name = name

    def generate_execution_plan(self, kg):
        return [self.double(node) for node in kg.nodes]

    async def transform(
        self, kg: KnowledgeGraph
    ) -> t.List[t.Tuple[Node, t.Tuple[str, t.Any]]]:
        filtered = self.filter(kg)
        nodes = sorted(
            filtered.nodes, key=lambda n: n.get_property("page_content") or ""
        )
        return [(node, await self.double(node)) for node in nodes]

    async def double(self, node):
        # Repeat the text in a single node's 'page_content' property
        content = node.get_property("page_content")
        if content is not None:
            node.properties["page_content"] = content * 2
        return node


@pytest.fixture
def kg():
    import string

    kg = KnowledgeGraph()
    for letter in string.ascii_uppercase[:10]:
        node = Node(
            properties={"page_content": letter},
            type=NodeType.DOCUMENT,
        )
        kg.add(node)
    return kg


def test_parallel_stores_transformations():
    t1 = DummyTransformation("A")
    t2 = DummyTransformation("B")
    p = Parallel(t1, t2)
    assert p.transformations == [t1, t2]


def test_parallel_generate_execution_plan_aggregates(kg):
    t1 = DummyTransformation("A")
    t2 = DummyTransformation("B")
    p = Parallel(t1, t2)
    coros = p.generate_execution_plan(kg)
    assert len(coros) == len(kg.nodes) * 2  # Each transformation runs on each node
    assert all(isinstance(c, types.CoroutineType) for c in coros)

    # Await all coroutines to avoid RuntimeWarning
    async def run_all():
        await asyncio.gather(*coros)

    asyncio.run(run_all())


def test_parallel_nested(kg):
    t1 = DummyTransformation("A")
    t2 = DummyTransformation("B")
    p_inner = Parallel(t1)
    p_outer = Parallel(p_inner, t2)
    coros = p_outer.generate_execution_plan(kg)
    assert len(coros) == len(kg.nodes) * 2  # Each transformation runs on each node
    assert all(isinstance(c, types.CoroutineType) for c in coros)

    # Await all coroutines to avoid RuntimeWarning
    async def run_all():
        await asyncio.gather(*coros)

    asyncio.run(run_all())


def test_get_desc_parallel_and_single():
    t1 = DummyTransformation("A")
    p = Parallel(t1)
    desc_p = get_desc(p)
    desc_t = get_desc(t1)
    assert "Parallel" not in desc_t
    assert "DummyTransformation" in desc_p or "DummyTransformation" in desc_t


def test_apply_transforms_single(kg):
    t1 = DummyTransformation()
    apply_transforms(kg, t1)
    # All nodes' page_content should be doubled
    for node in kg.nodes:
        content = node.get_property("page_content")
        assert content == (content[0] * 2)


def test_apply_transforms_list(kg):
    t1 = DummyTransformation()
    t2 = DummyTransformation()
    apply_transforms(kg, [t1, t2])
    # Each transformation doubles the content, so after two: x -> xxxx
    for node in kg.nodes:
        content = node.get_property("page_content")
        assert content == (content[0] * 2 * 2)


def test_apply_transforms_parallel(kg):
    t1 = DummyTransformation()
    t2 = DummyTransformation()
    p = Parallel(t1, t2)
    apply_transforms(kg, p)
    # Each transformation in parallel doubles the content, but both operate on the same initial state, so after both: x -> xx (not xxxx)
    for node in kg.nodes:
        content = node.get_property("page_content")
        assert content == (content[0] * 2 * 2)


def test_apply_transforms_invalid():
    kg = KnowledgeGraph()
    with pytest.raises(ValueError):
        apply_transforms(kg, 123)  # type: ignore
