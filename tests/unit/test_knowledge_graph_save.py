from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship


def test_knowledge_graph_save_with_problematic_chars(tmp_path):
    # Create a knowledge graph with special characters
    kg = KnowledgeGraph()

    # Create nodes with various Unicode characters including ones that might cause charmap codec issues
    problematic_chars = [
        chr(i) for i in range(0x0080, 0x00FF)  # Extended ASCII/Latin-1 characters
    ] + [
        "\u2022",  # bullet
        "\u2192",  # arrow
        "\u2665",  # heart
        "\u2605",  # star
        "\u221E",  # infinity
        "\u00B5",  # micro
        "\u2264",  # less than or equal
        "\u2265",  # greater than or equal
        "\u0391",  # Greek letters
        "\u0392",
        "\u0393",
        "\uFFFF",  # Special Unicode characters
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
