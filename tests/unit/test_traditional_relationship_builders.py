import asyncio
import copy
import math
import random
import string
from typing import List, Set, Tuple
from uuid import UUID

import numpy as np
import pytest

from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.transforms.relationship_builders.traditional import (
    JaccardSimilarityBuilder,
)


def generate_test_sets(
    n: int = 16,
    max_len: int = 32,
    min_similarity: float = 0.5,
    similar_fraction: float = 0.3,
) -> List[Set[str]]:
    """
    Generate `n` sets up to `max_len`, where at least `similar_fraction` of all possible
    pairs have Jaccard similarity >= `min_similarity`. The result is shuffled.

    Parameters:
    - n (int): Total number of sets to generate.
    - max_len (int): Maximum length of each set.
    - min_similarity (float): Minimum Jaccard similarity for similar pairs.
    - similar_fraction (float): Fraction (0-1) of sets that should be similar.

    Returns:
    - list: List of generated sets.
    """

    def generate_entity(k: int = 5) -> str:
        """Generate a random entity of length k."""
        return "".join(random.choices(string.ascii_lowercase, k=k))

    def jaccard(a: set[str], b: set[str]) -> float:
        from scipy.spatial.distance import jaccard as jaccard_dist

        # union of elements -> boolean indicator vectors
        elems = sorted(a | b)
        va = np.array([e in a for e in elems], dtype=bool)
        vb = np.array([e in b for e in elems], dtype=bool)
        # SciPy returns the Jaccard distance; similarity = 1 - distance
        return 1.0 - jaccard_dist(va, vb)

    # bias toward shorter lengths (expovariate with λ=1.0)
    def sample_length() -> int:
        length = int(random.expovariate(1.0))
        return min(length, max_len)

    total_pairs = n * (n - 1) // 2
    target_similar = math.ceil(total_pairs * similar_fraction)

    # Initialize all sets with random, ragged lengths
    sets = [{generate_entity() for _ in range(sample_length())} for _ in range(n)]

    # Count how many pairs are “similar” right now
    current_similar = len(jaccard_similarity_pair(sets, min_similarity))

    # Iteratively fix random non‐similar pairs until we hit target
    max_attempts = target_similar * 10
    attempts = 0

    while current_similar < target_similar and attempts < max_attempts:
        # pick a non‐similar pair
        bad_pairs = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if jaccard(sets[i], sets[j]) < min_similarity
        ]
        if not bad_pairs:
            break
        i, j = random.choice(bad_pairs)

        # decide new lengths
        Li, Lj = sample_length(), sample_length()
        # solve for needed intersection size intersection_size such that
        #   intersection_size / (Li + Lj - intersection_size) >= min_similarity
        intersection_size = math.ceil(min_similarity * (Li + Lj) / (1 + min_similarity))

        # build new similar pair
        shared = {generate_entity() for _ in range(intersection_size)}
        Ai = shared | {generate_entity() for _ in range(Li - intersection_size)}
        Bj = shared | {generate_entity() for _ in range(Lj - intersection_size)}

        sets[i], sets[j] = Ai, Bj

        current_similar = len(jaccard_similarity_pair(sets, min_similarity))
        attempts += 1
    else:
        raise ValueError(
            f"Could not generate enough similar pairs after {max_attempts} attempts."
        )

    # Create a core set of shared elements for similar sets
    core_size = max(1, int(max_len * min_similarity))
    core = {generate_entity() for _ in range(core_size)}

    # Create a set of unique elements to draw from
    base_pool = {generate_entity() for _ in range(n * max_len * 8)}
    base_pool -= core

    n_similar = int(n * similar_fraction)
    n_dissimilar = n - n_similar

    # Pre-calculate max add'l unique elements that can be added to core while still guaranteeing min_similarity
    max_unique = int(core_size * ((1 - min_similarity) / min_similarity))
    if max_unique > max_len:
        raise ValueError(
            "max_unique exceeds max_len, cannot guarantee min_similarity with given parameters."
        )

    # Generate similar sets
    similar = []
    for _ in range(n_similar):
        # Random size for this set, at least the core size
        set_len = core_size + random.randint(0, max_unique)
        s = core.copy()
        # Add random elements from the base pool until we reach set_len
        while len(s) < set_len:
            if not base_pool:
                raise ValueError("Base pool is empty, cannot generate more sets.")
            element = base_pool.pop()
            if element not in s:
                s.add(element)
        similar.append(s)

    # Generate dissimilar sets
    dissimilar = []
    for _ in range(n_dissimilar):
        set_len = random.randint(0, max_len)
        s = set()
        while len(s) < set_len:
            if not base_pool:
                raise ValueError("Base pool is empty, cannot generate more sets.")
            element = base_pool.pop()
            if element not in s:
                s.add(element)
        dissimilar.append(s)

    sets = similar + dissimilar
    random.shuffle(sets)
    return sets


def validate_sets(sets: list[set[str]], min_similarity: float, similar_fraction: float):
    n = len(sets)
    n_similar_needed = int(n * similar_fraction)

    similar_pairs = jaccard_similarity_pair(sets, min_similarity)
    n_similar_pairs = len(similar_pairs)
    actual_similar_fraction = n_similar_pairs / (n * (n - 1) // 2)

    print(f"Expected similar pairs: {n_similar_needed}")
    print(f"Actual similar pairs: {n_similar_pairs}")
    print(f"Actual similar fraction: {actual_similar_fraction:.2f}")
    print(f"Similarity threshold: {min_similarity}")


def jaccard_similarity_matrix(sets: List[Set[str]]) -> np.ndarray:
    """Calculate Jaccard similarity matrix for a list of string sets."""
    n = len(sets)
    similarity = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            intersection = sets[i].intersection(sets[j])
            union = sets[i].union(sets[j])
            score = len(intersection) / len(union) if union else 0.0
            similarity[i, j] = similarity[j, i] = score

    return similarity


def jaccard_similarity_pair(
    sets: List[Set[str]], threshold: float
) -> List[Tuple[int, int, float]]:
    """Find pairs of sets with Jaccard similarity >= threshold."""
    similarity_matrix = jaccard_similarity_matrix(sets)
    similar_pairs = np.argwhere(similarity_matrix >= threshold)

    return [
        (int(i), int(j), float(similarity_matrix[i, j]))
        for i, j in similar_pairs
        if i < j  # avoid self-pairs and duplicates
    ]


@pytest.fixture
def simple_kg():
    # Arrange: create a simple knowledge graph with embeddings
    # roughly, we expect the following relationships:
    # 1 <-> 2 (0.0 similarity)
    # 2 <-> 3 (0.1667 similarity)
    # 1 <-> 3 (0.25 similarity)
    nodes = [
        Node(
            id=UUID("4da47a69-539c-49a2-b289-01780989d82c"),
            type=NodeType.DOCUMENT,
            properties={
                "entities": {"cat", "dog", "fish", "fox", "bird"},
            },
        ),
        Node(
            id=UUID("f353e5c2-e432-4d1e-84a8-d750c93d4edf"),
            type=NodeType.DOCUMENT,
            properties={
                "entities": {"apple", "banana"},
            },
        ),
        Node(
            id=UUID("437c8c08-cef6-4ebf-a35f-93d6168b61a4"),
            type=NodeType.DOCUMENT,
            properties={
                "entities": {"cat", "banana", "dog", "rock", "tree"},
            },
        ),
    ]
    return KnowledgeGraph(nodes=nodes)


# node order
# UUID("4da47a69-539c-49a2-b289-01780989d82c")
# UUID("f353e5c2-e432-4d1e-84a8-d750c93d4edf")
# UUID("437c8c08-cef6-4ebf-a35f-93d6168b61a4")


@pytest.mark.parametrize(
    "n_test_sets, max_len, threshold",
    [
        (8, 100, 0.2),
        (16, 8, 0.1),
        (16, 16, 0.5),
        (32, 5, 0.3),
    ],
)
def test__find_similar_embedding_pairs_jaccard(n_test_sets, max_len, threshold):
    """
    Validate that _find_similar_embedding_pairs correctly identifies pairs when compared with scipy's jaccard distance.
    """
    sets = generate_test_sets(
        n=n_test_sets,
        max_len=max_len,
        min_similarity=min(threshold + 0.05, 1.0),
        similar_fraction=0.3,
    )
    expected = jaccard_similarity_pair(sets, threshold)

    kg = KnowledgeGraph(
        nodes=[Node(type=NodeType.DOCUMENT, properties={"entities": s}) for s in sets]
    )
    builder = JaccardSimilarityBuilder(property_name="entities", threshold=threshold)
    result = list(asyncio.run(builder._find_similar_embedding_pairs(kg)))

    assert len(result) == len(expected)
    for i, j, similarity_float in result:
        assert i < j, "Pairs should be ordered (i < j)"
        assert (
            similarity_float >= threshold
        ), f"Similarity {similarity_float} should be >= {threshold}"
        for x, y, expected_similarity in expected:
            if i == x and j == y:
                assert similarity_float == pytest.approx(expected_similarity)
                break


class TestJaccardSimilarityBuilder:
    @pytest.mark.asyncio
    async def test_no_self_similarity_relationships(self, simple_kg):
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.1)
        relationships = await builder.transform(copy.deepcopy(simple_kg))
        for r in relationships:
            assert (
                r.source.id != r.target.id
            ), "Self-relationships should not be created"

    @pytest.mark.asyncio
    async def test_no_duplicate_relationships(self, simple_kg):
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.1)
        relationships = await builder.transform(copy.deepcopy(simple_kg))
        seen = set()
        for r in relationships:
            pair = tuple(sorted([r.source.id, r.target.id]))
            assert pair not in seen, "Duplicate relationships found"
            seen.add(pair)

    @pytest.mark.asyncio
    async def test_similarity_at_threshold(self):
        node1 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        node2 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        kg = KnowledgeGraph(nodes=[node1, node2])
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=1.0)
        relationships = await builder.transform(kg)
        assert len(relationships) == 1, "Should create relationship at threshold"

    @pytest.mark.asyncio
    async def test_all_below_threshold(self):
        node1 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        node2 = Node(type=NodeType.DOCUMENT, properties={"entities": {"x", "y", "z"}})
        kg = KnowledgeGraph(nodes=[node1, node2])
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.1)
        relationships = await builder.transform(kg)
        assert (
            len(relationships) == 0
        ), "No relationships should be created below threshold"

    @pytest.mark.asyncio
    async def test_all_above_threshold(self):
        node1 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        node2 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        node3 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        kg = KnowledgeGraph(nodes=[node1, node2, node3])
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.9)
        relationships = await builder.transform(kg)
        assert len(relationships) == 3

    @pytest.mark.asyncio
    async def test_malformed_entities_raises(self):
        node1 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        node2 = Node(type=NodeType.DOCUMENT, properties={"entities": None})
        kg = KnowledgeGraph(nodes=[node1, node2])
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.5)
        with pytest.raises(ValueError):
            await builder.transform(kg)

    @pytest.mark.asyncio
    async def test_jaccard_similarity_builder_empty_graph(self):
        kg = KnowledgeGraph(nodes=[])
        builder = JaccardSimilarityBuilder(property_name="entities")
        relationships = await builder.transform(kg)
        assert relationships == []

    @pytest.mark.asyncio
    async def test_jaccard_similarity_builder_basic(self, simple_kg):
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.15)
        relationships = await builder.transform(simple_kg)
        assert all(isinstance(r, Relationship) for r in relationships)
        assert all(r.type == "jaccard_similarity" for r in relationships)
        # 2 <-> 3 (~0.1667 similarity)
        assert any(
            str(r.source.id) == "f353e5c2-e432-4d1e-84a8-d750c93d4edf"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in relationships
        )
        # 1 <-> 3 (~0.25 similarity)
        assert any(
            str(r.source.id) == "4da47a69-539c-49a2-b289-01780989d82c"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in relationships
        )

    @pytest.mark.asyncio
    async def test_jaccard_similarity_builder_no_entities(self):
        kg = KnowledgeGraph(
            nodes=[
                Node(type=NodeType.DOCUMENT, properties={}),
                Node(type=NodeType.DOCUMENT, properties={}),
            ]
        )
        builder = JaccardSimilarityBuilder(property_name="entities")
        with pytest.raises(ValueError, match="has no entities"):
            await builder.transform(kg)

    @pytest.mark.asyncio
    async def test_apply_transforms_cosine_similarity_builder(self, simple_kg):
        from ragas.run_config import RunConfig
        from ragas.testset.transforms.engine import apply_transforms

        # JaccardSimilarityBuilder should add relationships to the graph
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.15)
        kg = simple_kg
        # Should mutate kg in-place
        apply_transforms(kg, builder, run_config=RunConfig(max_workers=2))
        # Check that relationships were added
        assert any(
            r.type == "jaccard_similarity" for r in kg.relationships
        ), "No jaccard_similarity relationships found after apply_transforms"
        # Check that expected relationship exists
        assert any(
            str(r.source.id) == "f353e5c2-e432-4d1e-84a8-d750c93d4edf"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in kg.relationships
        )
        # 1 <-> 3 (~0.8258 similarity)
        assert any(
            str(r.source.id) == "4da47a69-539c-49a2-b289-01780989d82c"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in kg.relationships
        )
