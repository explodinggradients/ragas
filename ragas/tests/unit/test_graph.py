import pytest

from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship


def test_knowledge_graph_save_with_problematic_chars(tmp_path):
    # Create a knowledge graph with special characters
    kg = KnowledgeGraph()

    # Create nodes with various Unicode characters including ones that might cause charmap codec issues
    problematic_chars = [
        chr(i)
        for i in range(0x0080, 0x00FF)  # Extended ASCII/Latin-1 characters
    ] + [
        "\u2022",  # bullet
        "\u2192",  # arrow
        "\u2665",  # heart
        "\u2605",  # star
        "\u221e",  # infinity
        "\u00b5",  # micro
        "\u2264",  # less than or equal
        "\u2265",  # greater than or equal
        "\u0391",  # Greek letters
        "\u0392",
        "\u0393",
        "\uffff",  # Special Unicode characters
    ]

    # Create multiple nodes with combinations of problematic characters
    for i, char in enumerate(problematic_chars):
        text = f"Test{char}Text with special char at position {i}"
        node = Node(
            properties={
                "text": text,
                "description": f"Node {i} with {char}",
                "metadata": f"Extra {char} info",
            },
            type=NodeType.CHUNK,
        )
        kg.add(node)

    # Add some relationships to make it more realistic
    nodes = kg.nodes
    for i in range(len(nodes) - 1):
        rel = Relationship(
            source=nodes[i],
            target=nodes[i + 1],
            type="next",
            properties={"info": f"Link {i} with special char {problematic_chars[i]}"},
        )
        kg.add(rel)

    # Try to save to a temporary file
    save_path = tmp_path / "test_knowledge_graph.json"
    kg.save(str(save_path))

    # Try to load it back to verify
    loaded_kg = KnowledgeGraph.load(str(save_path))

    # Verify the content was preserved
    assert len(loaded_kg.nodes) == len(kg.nodes)
    assert len(loaded_kg.relationships) == len(kg.relationships)

    # Verify the special characters were preserved in the first node
    assert loaded_kg.nodes[0].properties["text"] == nodes[0].properties["text"]


class TestFindIndirectClusters:
    # Helper function to compare lists of sets
    def assert_sets_equal(self, list1, list2):
        """Asserts that two lists of sets are equal, ignoring order."""
        set1_of_frozensets = {frozenset(s) for s in list1}
        set2_of_frozensets = {frozenset(s) for s in list2}
        assert set1_of_frozensets == set2_of_frozensets

    @pytest.fixture
    def simple_graph(self):
        """
        Provides a simple graph for testing.
        Structure:
        A -> B -> C
        A -> D
        """
        kg = KnowledgeGraph()
        node_a = Node(properties={"id": "A"})
        node_b = Node(properties={"id": "B"})
        node_c = Node(properties={"id": "C"})
        node_d = Node(properties={"id": "D"})

        nodes = [node_a, node_b, node_c, node_d]
        for n in nodes:
            kg.add(n)

        kg.add(Relationship(source=node_a, target=node_b, type="link"))
        kg.add(Relationship(source=node_b, target=node_c, type="link"))
        kg.add(Relationship(source=node_a, target=node_d, type="link"))

        return kg, {"A": node_a, "B": node_b, "C": node_c, "D": node_d}

    def test_simple_paths(self, simple_graph):
        # Arrange
        kg, nodes = simple_graph
        node_a, node_b, node_c, node_d = (
            nodes["A"],
            nodes["B"],
            nodes["C"],
            nodes["D"],
        )

        # Act
        clusters = kg.find_indirect_clusters()

        # Assert
        expected_clusters = [
            {node_a, node_b},
            {node_a, node_b, node_c},
            {node_b, node_c},
            {node_a, node_d},
        ]
        self.assert_sets_equal(clusters, expected_clusters)

    def test_with_depth_limit(self, simple_graph):
        # Arrange
        kg, nodes = simple_graph
        node_a, node_b, node_d = nodes["A"], nodes["B"], nodes["D"]

        # Act
        # depth_limit=1 means paths of length 1 (2 nodes)
        clusters = kg.find_indirect_clusters(depth_limit=1)

        # Assert
        # Should only find paths of length 1 (A->B, B->C, A->D)
        # The implementation also includes sub-paths, so we expect {A,B}, {B,C}, {A,D}
        # The current implementation's depth logic is slightly complex.
        # A path of length 1 has 2 nodes. `len(path)` will be 2.
        # The check is `len(path) > depth_limit`. So `2 > 1` is true.
        # It adds the path of 2 nodes, then stops exploring from there.
        expected_clusters = [
            {node_a, node_b},
            {nodes["B"], nodes["C"]},
            {node_a, node_d},
        ]
        self.assert_sets_equal(clusters, expected_clusters)

    def test_with_cycle(self):
        # Arrange
        kg = KnowledgeGraph()
        node_a = Node(properties={"id": "A"})
        node_b = Node(properties={"id": "B"})
        node_c = Node(properties={"id": "C"})
        nodes = [node_a, node_b, node_c]
        for n in nodes:
            kg.add(n)

        # A -> B -> C -> A
        kg.add(Relationship(source=node_a, target=node_b, type="link"))
        kg.add(Relationship(source=node_b, target=node_c, type="link"))
        kg.add(Relationship(source=node_c, target=node_a, type="link"))

        # Act
        clusters = kg.find_indirect_clusters(depth_limit=3)

        # Assert
        # The method should not get stuck in an infinite loop and should find unique paths up to the depth limit.
        expected_clusters = [
            {node_a, node_b},
            {node_a, node_b, node_c},
            {node_b, node_c},
            {node_b, node_c, node_a},  # same as {a,b,c}
            {node_c, node_a},
            {node_c, node_a, node_b},  # same as {a,b,c}
        ]
        self.assert_sets_equal(clusters, expected_clusters)

    def test_bidirectional(self):
        # Arrange
        kg = KnowledgeGraph()
        node_a = Node(properties={"id": "A"})
        node_b = Node(properties={"id": "B"})
        kg.add(node_a)
        kg.add(node_b)
        kg.add(
            Relationship(source=node_a, target=node_b, type="link", bidirectional=True)
        )

        # Act
        clusters = kg.find_indirect_clusters()

        # Assert
        # A <-> B should be found from both directions, but deduplicated.
        expected_clusters = [{node_a, node_b}]
        self.assert_sets_equal(clusters, expected_clusters)

    def test_no_valid_paths(self):
        # Arrange
        kg = KnowledgeGraph()
        kg.add(Node(properties={"id": "A"}))
        kg.add(Node(properties={"id": "B"}))

        # Act
        clusters = kg.find_indirect_clusters()

        # Assert
        assert clusters == []

    def test_relationship_condition(self):
        # Arrange
        kg = KnowledgeGraph()
        node_a = Node(properties={"id": "A"})
        node_b = Node(properties={"id": "B"})
        node_c = Node(properties={"id": "C"})
        nodes = [node_a, node_b, node_c]
        for n in nodes:
            kg.add(n)

        kg.add(Relationship(source=node_a, target=node_b, type="allowed"))
        kg.add(Relationship(source=node_b, target=node_c, type="blocked"))

        # Act
        clusters = kg.find_indirect_clusters(
            relationship_condition=lambda r: r.type == "allowed"
        )

        # Assert
        # Should only find the path A->B, as B->C is blocked by the condition.
        expected_clusters = [{node_a, node_b}]
        self.assert_sets_equal(clusters, expected_clusters)

    def test_disconnected_components(self):
        # Arrange
        kg = KnowledgeGraph()
        node_a = Node(properties={"id": "A"})
        node_b = Node(properties={"id": "B"})
        node_x = Node(properties={"id": "X"})
        node_y = Node(properties={"id": "Y"})
        nodes = [node_a, node_b, node_x, node_y]
        for n in nodes:
            kg.add(n)

        # Component 1: A -> B
        kg.add(Relationship(source=node_a, target=node_b, type="link"))
        # Component 2: X -> Y
        kg.add(Relationship(source=node_x, target=node_y, type="link"))

        # Act
        clusters = kg.find_indirect_clusters()

        # Assert
        expected_clusters = [
            {node_a, node_b},
            {node_x, node_y},
        ]
        self.assert_sets_equal(clusters, expected_clusters)
