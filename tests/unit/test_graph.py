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
        Separate triangle: E-F-G-E (3-clique)
        4-clique: D-E-F-G (all connected)
        """
        kg = KnowledgeGraph()
        node_a = Node(properties={"id": "A"})
        node_b = Node(properties={"id": "B"})
        node_c = Node(properties={"id": "C"})
        node_d = Node(properties={"id": "D"})
        node_e = Node(properties={"id": "E"})
        node_f = Node(properties={"id": "F"})
        node_g = Node(properties={"id": "G"})

        nodes = [node_a, node_b, node_c, node_d, node_e, node_f, node_g]
        for n in nodes:
            kg.add(n)

        # Triangle 1: A-B-C-A (3-clique)
        kg.add(Relationship(source=node_a, target=node_b, type="link"))
        kg.add(Relationship(source=node_b, target=node_c, type="link"))
        kg.add(Relationship(source=node_c, target=node_a, type="link"))

        # Add D to make a 4-clique A-B-C-D
        kg.add(
            Relationship(source=node_a, target=node_d, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_b, target=node_d, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_c, target=node_d, type="link", bidirectional=True)
        )

        # Separate triangle: E-F-G-E (3-clique)
        kg.add(Relationship(source=node_e, target=node_f, type="link"))
        kg.add(Relationship(source=node_f, target=node_g, type="link"))
        kg.add(Relationship(source=node_g, target=node_e, type="link"))

        # Add D to make a 4-clique E-F-G-D
        kg.add(
            Relationship(source=node_e, target=node_d, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_f, target=node_d, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_g, target=node_d, type="link", bidirectional=True)
        )

        return kg, {
            "A": node_a,
            "B": node_b,
            "C": node_c,
            "D": node_d,
            "E": node_e,
            "F": node_f,
            "G": node_g,
        }

    # Should find 2 clusters - a/b/c and e/f/g; d should drop out since it is involved in both
    @pytest.mark.parametrize(
        "depth_limit,expected_cluster_types",
        [
            (
                2,
                [
                    # depth_limit=2 allows paths up to length 2 (3 nodes)
                    ("A", "B"),
                    ("A", "C"),
                    ("B", "C"),
                    ("A", "B", "C"),
                    ("E", "F"),
                    ("E", "G"),
                    ("F", "G"),
                    ("E", "F", "G"),
                ],
            ),
            (
                3,
                [
                    # depth_limit=3 allows paths up to length 3 (4 nodes)
                    # but we don't have any paths that long in the simple graph
                    ("A", "B"),
                    ("A", "C"),
                    ("B", "C"),
                    ("A", "B", "C"),
                    ("E", "F"),
                    ("E", "G"),
                    ("F", "G"),
                    ("E", "F", "G"),
                ],
            ),
            (
                4,
                [],  # depth_limit=4 > max(cluster_size), so no paths are identified
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

        # print(f"\n=== Depth Limit {depth_limit} ===")
        # print(f"Found {len(clusters)} clusters, expected {len(expected_clusters)}")

        # # Helper function to get node names from a cluster
        # def get_cluster_names(cluster):
        #     return sorted(
        #         [node.properties.get("id", str(node.id)[:6]) for node in cluster]
        #     )

        # print("\nFound clusters:")
        # for i, cluster in enumerate(
        #     sorted(clusters, key=lambda c: (len(c), get_cluster_names(c)))
        # ):
        #     names = get_cluster_names(cluster)
        #     print(f"  {i + 1}. {{{', '.join(names)}}}")

        # print("\nExpected clusters:")
        # for i, cluster in enumerate(
        #     sorted(expected_clusters, key=lambda c: (len(c), get_cluster_names(c)))
        # ):
        #     names = get_cluster_names(cluster)
        #     print(f"  {i + 1}. {{{', '.join(names)}}}")

        # # Show differences if any
        # found_sets = {frozenset(get_cluster_names(c)) for c in clusters}
        # expected_sets = {frozenset(get_cluster_names(c)) for c in expected_clusters}

        # if found_sets != expected_sets:
        #     missing = expected_sets - found_sets
        #     extra = found_sets - expected_sets
        #     if missing:
        #         print(f"\nMissing clusters: {[set(s) for s in missing]}")
        #     if extra:
        #         print(f"Extra clusters: {[set(s) for s in extra]}")
        # else:
        #     print("\n✓ All clusters match!")
        # print("=" * 40)

        self.assert_sets_equal(clusters, expected_clusters)

    def test_with_cycle(self, simple_graph):
        # above test_with_depth_limit uses simple_graph which already has cycles
        pass

    def test_bidirectional(self):
        """Test that bidirectional relationships are handled correctly.
        Since relationships are filtered by type, we can assume that all relationships will be bidirectional
        """
        # Arrange - Use the simple_graph and add a bidirectional relationship
        kg = KnowledgeGraph()
        node_a = Node(properties={"id": "A"})
        node_b = Node(properties={"id": "B"})
        node_c = Node(properties={"id": "C"})
        node_d = Node(properties={"id": "D"})
        node_e = Node(properties={"id": "E"})
        node_f = Node(properties={"id": "F"})
        node_g = Node(properties={"id": "G"})
        node_h = Node(properties={"id": "H"})

        nodes = [node_a, node_b, node_c, node_d, node_e, node_f, node_g, node_h]
        for n in nodes:
            kg.add(n)

        kg.add(
            Relationship(source=node_a, target=node_b, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_b, target=node_c, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_c, target=node_d, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_d, target=node_a, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_a, target=node_c, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_b, target=node_d, type="link", bidirectional=True)
        )

        kg.add(
            Relationship(source=node_e, target=node_f, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_f, target=node_g, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_g, target=node_h, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_h, target=node_e, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_e, target=node_g, type="link", bidirectional=True)
        )
        kg.add(
            Relationship(source=node_f, target=node_h, type="link", bidirectional=True)
        )

        # Act
        clusters = kg.find_indirect_clusters()

        # Assert
        expected_clusters = [
            {node_a, node_b},
            {node_a, node_c},
            {node_a, node_d},
            {node_b, node_c},
            {node_b, node_d},
            {node_c, node_d},
            {node_a, node_b, node_c},
            {node_a, node_b, node_d},
            {node_a, node_c, node_d},
            {node_b, node_c, node_d},
            {node_a, node_b, node_c, node_d},
            {node_e, node_f},
            {node_e, node_g},
            {node_e, node_h},
            {node_f, node_g},
            {node_f, node_h},
            {node_g, node_h},
            {node_e, node_f, node_g},
            {node_e, node_f, node_h},
            {node_e, node_g, node_h},
            {node_f, node_g, node_h},
            {node_e, node_f, node_g, node_h},
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

        # Cycle: A-B-C-A
        #          \D/
        kg.add(Relationship(source=node_a, target=node_b, type="link"))
        kg.add(Relationship(source=node_b, target=node_c, type="link"))
        kg.add(Relationship(source=node_c, target=node_a, type="link"))

        kg.add(Relationship(source=node_b, target=node_d, type="link"))
        kg.add(Relationship(source=node_c, target=node_d, type="link"))
        kg.add(Relationship(source=node_d, target=node_a, type="link"))

        # Act
        clusters_connected = kg.find_indirect_clusters(
            relationship_condition=lambda r: r.type == "link"
        )

        kg.remove_node(node_d)
        kg.add(node_d)
        kg.add(Relationship(source=node_b, target=node_d, type="link"))
        kg.add(Relationship(source=node_c, target=node_d, type="link"))
        kg.add(Relationship(source=node_d, target=node_a, type="broken"))

        clusters_broken = kg.find_indirect_clusters(
            relationship_condition=lambda r: r.type == "link"
        )

        # Assert
        expected_clusters = [
            {node_a, node_b},
            {node_a, node_c},
            {node_b, node_c},
            {node_a, node_b, node_c},
        ]

        # Should only find clusters using "link" relationships, excluding "blocked" ones
        assert len(clusters_connected) != len(clusters_broken)
        self.assert_sets_equal(clusters_broken, expected_clusters)

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
            {node_a, node_b},
            {node_a, node_c},
            {node_b, node_c},
            {node_a, node_b, node_c},
            {node_x, node_y},
            {node_x, node_z},
            {node_y, node_z},
            {node_x, node_y, node_z},
        ]
        self.assert_sets_equal(clusters, expected_clusters)
