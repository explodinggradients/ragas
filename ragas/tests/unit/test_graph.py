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
        Triangle: A-B-C-A (3-clique)
        4-clique: A-B-C-D (all connected)
        Separate triangle: D-E-F-D (3-clique)
        """
        kg = KnowledgeGraph()
        node_a = Node(properties={"id": "A"})
        node_b = Node(properties={"id": "B"})
        node_c = Node(properties={"id": "C"})
        node_d = Node(properties={"id": "D"})
        node_e = Node(properties={"id": "E"})
        node_f = Node(properties={"id": "F"})

        nodes = [node_a, node_b, node_c, node_d, node_e, node_f]
        for n in nodes:
            kg.add(n)

        # Triangle 1: A-B-C-A (3-clique)
        kg.add(Relationship(source=node_a, target=node_b, type="link"))
        kg.add(Relationship(source=node_b, target=node_c, type="link"))
        kg.add(Relationship(source=node_c, target=node_a, type="link"))

        # Add D to make a 4-clique A-B-C-D
        kg.add(Relationship(source=node_a, target=node_d, type="link"))
        kg.add(Relationship(source=node_b, target=node_d, type="link"))
        kg.add(Relationship(source=node_c, target=node_d, type="link"))

        # Separate triangle: D-E-F-D (3-clique)
        kg.add(Relationship(source=node_d, target=node_e, type="link"))
        kg.add(Relationship(source=node_e, target=node_f, type="link"))
        kg.add(Relationship(source=node_f, target=node_d, type="link"))

        return kg, {
            "A": node_a,
            "B": node_b,
            "C": node_c,
            "D": node_d,
            "E": node_e,
            "F": node_f,
        }

    @pytest.mark.parametrize(
        "depth_limit,expected_cluster_types",
        [
            (
                2,
                [
                    # depth_limit=2 allows paths up to length 2 (3 nodes)
                    # Should find all edges (2-node clusters) and triangles (3-node clusters)
                    # Edges from 4-clique A-B-C-D
                    ("A", "B"),
                    ("A", "C"),
                    ("A", "D"),
                    ("B", "C"),
                    ("B", "D"),
                    ("C", "D"),
                    # Triangles from 4-clique A-B-C-D
                    ("A", "B", "C"),
                    ("A", "B", "D"),
                    ("A", "C", "D"),
                    ("B", "C", "D"),
                    # Edges from 3-clique D-E-F
                    ("D", "E"),
                    ("D", "F"),
                    ("E", "F"),
                    # Triangle from 3-clique D-E-F
                    ("D", "E", "F"),
                ],
            ),
            (
                3,
                [
                    # depth_limit=3 allows paths up to length 3 (4 nodes)
                    # Should find all previous clusters plus the full 4-clique
                    # Edges from 4-clique A-B-C-D
                    ("A", "B"),
                    ("A", "C"),
                    ("A", "D"),
                    ("B", "C"),
                    ("B", "D"),
                    ("C", "D"),
                    # Triangles from 4-clique A-B-C-D
                    ("A", "B", "C"),
                    ("A", "B", "D"),
                    ("A", "C", "D"),
                    ("B", "C", "D"),
                    # Full 4-clique A-B-C-D
                    ("A", "B", "C", "D"),
                    # Edges from 3-clique D-E-F
                    ("D", "E"),
                    ("D", "F"),
                    ("E", "F"),
                    # Triangle from 3-clique D-E-F
                    ("D", "E", "F"),
                ],
            ),
            (
                4,
                [
                    # depth_limit=4 allows paths up to length 4 (5 nodes)
                    # Since our largest clique is 4 nodes, this should be same as depth_limit=3
                    # but might include some longer paths through the bridge node D
                    # Edges from 4-clique A-B-C-D
                    ("A", "B"),
                    ("A", "C"),
                    ("A", "D"),
                    ("B", "C"),
                    ("B", "D"),
                    ("C", "D"),
                    # Triangles from 4-clique A-B-C-D
                    ("A", "B", "C"),
                    ("A", "B", "D"),
                    ("A", "C", "D"),
                    ("B", "C", "D"),
                    # Full 4-clique A-B-C-D
                    ("A", "B", "C", "D"),
                    # Edges from 3-clique D-E-F
                    ("D", "E"),
                    ("D", "F"),
                    ("E", "F"),
                    # Triangle from 3-clique D-E-F
                    ("D", "E", "F"),
                ],
            ),
        ],
    )
    def test_with_depth_limit(self, simple_graph, depth_limit, expected_cluster_types):
        # Arrange
        kg, nodes = simple_graph

        # Act
        clusters = kg.find_indirect_clusters(depth_limit=depth_limit)

        # Assert
        # Convert expected cluster types (node IDs) to actual node sets
        expected_clusters = [
            {nodes[node_id] for node_id in cluster_tuple}
            for cluster_tuple in expected_cluster_types
        ]

        self.assert_sets_equal(clusters, expected_clusters)

    def test_with_cycle(self, simple_graph):
        # above test_with_depth_limit uses simple_graph which already has cycles
        pass

    def test_bidirectional(self, simple_graph):
        # Arrange - Use the simple_graph and add a bidirectional relationship
        kg, nodes = simple_graph
        node_a, node_b, node_c, node_d, node_e, node_f = (
            nodes["A"],
            nodes["B"],
            nodes["C"],
            nodes["D"],
            nodes["E"],
            nodes["F"],
        )

        # Add an additional bidirectional relationship to test that feature
        node_g = Node(properties={"id": "G"})
        node_h = Node(properties={"id": "H"})
        node_i = Node(properties={"id": "I"})
        kg.add(node_g)
        kg.add(node_h)
        kg.add(node_i)

        # Create a triangle with bidirectional relationships
        kg.add(
            Relationship(source=node_g, target=node_h, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_h, target=node_i, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_i, target=node_g, type="link", bidirectional=True)
        )

        # Act
        clusters = kg.find_indirect_clusters()

        # Assert
        # Should find all clusters from the original simple_graph plus the new triangle G-H-I
        expected_clusters = [
            # Edges from 4-clique A-B-C-D
            {node_a, node_b},
            {node_a, node_c},
            {node_a, node_d},
            {node_b, node_c},
            {node_b, node_d},
            {node_c, node_d},
            # Triangles from 4-clique A-B-C-D
            {node_a, node_b, node_c},
            {node_a, node_b, node_d},
            {node_a, node_c, node_d},
            {node_b, node_c, node_d},
            # Full 4-clique A-B-C-D
            {node_a, node_b, node_c, node_d},
            # Edges from 3-clique D-E-F
            {node_d, node_e},
            {node_d, node_f},
            {node_e, node_f},
            # Triangle from 3-clique D-E-F
            {node_d, node_e, node_f},
            # Edges from new triangle G-H-I
            {node_g, node_h},
            {node_g, node_i},
            {node_h, node_i},
            # Triangle from new triangle G-H-I
            {node_g, node_h, node_i},
        ]
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
        node_d = Node(properties={"id": "D"})

        nodes = [node_a, node_b, node_c, node_d]
        for n in nodes:
            kg.add(n)

        # Triangle 1: A-B-C-A (3-clique)
        kg.add(Relationship(source=node_a, target=node_b, type="link"))
        kg.add(Relationship(source=node_b, target=node_c, type="link"))
        kg.add(Relationship(source=node_c, target=node_a, type="link"))

        # Add D to make a 4-clique A-B-C-D
        kg.add(Relationship(source=node_a, target=node_d, type="link"))
        kg.add(Relationship(source=node_b, target=node_d, type="blocked"))
        kg.add(Relationship(source=node_c, target=node_d, type="link"))

        # Act
        clusters = kg.find_indirect_clusters(
            relationship_condition=lambda r: r.type == "link"
        )

        # Assert
        # Should only find clusters using "link" relationships, excluding "blocked" ones
        # Since D-B relationship is blocked, we won't have the full 4-clique
        expected_clusters = [
            # Edges from 4-clique A-B-C-D
            {node_a, node_b},
            {node_a, node_c},
            {node_a, node_d},
            {node_b, node_c},
            # {node_b, node_d}, # broken
            {node_c, node_d},
            # Triangles from 4-clique A-B-C-D
            {node_a, node_b, node_c},
            # {node_a, node_b, node_d}, # broken
            {node_a, node_c, node_d},
            {node_b, node_c, node_d},
            # Full 4-clique A-B-C-D
            {node_a, node_b, node_c, node_d},
        ]
        self.assert_sets_equal(clusters, expected_clusters)

    def test_disconnected_components(self):
        # Arrange - Create multiple disconnected triangles (3-cliques)
        kg = KnowledgeGraph()

        # Component 1: Triangle A-B-C
        node_a = Node(properties={"id": "A"})
        node_b = Node(properties={"id": "B"})
        node_c = Node(properties={"id": "C"})
        kg.add(node_a)
        kg.add(node_b)
        kg.add(node_c)
        kg.add(Relationship(source=node_a, target=node_b, type="link"))
        kg.add(Relationship(source=node_b, target=node_c, type="link"))
        kg.add(Relationship(source=node_c, target=node_a, type="link"))

        # Component 2: Triangle X-Y-Z
        node_x = Node(properties={"id": "X"})
        node_y = Node(properties={"id": "Y"})
        node_z = Node(properties={"id": "Z"})
        kg.add(node_x)
        kg.add(node_y)
        kg.add(node_z)
        kg.add(Relationship(source=node_x, target=node_y, type="link"))
        kg.add(Relationship(source=node_y, target=node_z, type="link"))
        kg.add(Relationship(source=node_z, target=node_x, type="link"))

        # Act
        clusters = kg.find_indirect_clusters()

        # Assert
        # Should find two separate triangular clusters
        expected_clusters = [
            # Edges from triangle A-B-C
            {node_a, node_b},
            {node_a, node_c},
            {node_b, node_c},
            # Triangle A-B-C
            {node_a, node_b, node_c},
            # Edges from triangle X-Y-Z
            {node_x, node_y},
            {node_x, node_z},
            {node_y, node_z},
            # Triangle X-Y-Z
            {node_x, node_y, node_z},
        ]
        self.assert_sets_equal(clusters, expected_clusters)
