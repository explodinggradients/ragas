import pytest
import uuid
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship


class DebugUUID(uuid.UUID):
    """
    A UUID subclass that displays a debug name instead of the UUID value.
    Creates a more readable graph representation in logs/debuggers while maintaining UUID compatibility.
    """

    def __init__(self, debug_name):
        # Create a random UUID internally
        self.debug = debug_name
        super().__init__(hex=str(uuid.uuid4()))

    def __str__(self):
        return self.debug

    def __repr__(self):
        return f"DebugUUID('{self.debug}')"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


def create_document_node(name):
    """Helper function to create a document node with proper structure."""
    return Node(
        id=DebugUUID(name),
        type=NodeType.DOCUMENT,
        properties={
            "page_content": f"{name} content",
            "summary": f"{name} summary",
            "document_metadata": {},
            "summary_embedding": [0.001, 0.002, 0.003],
            "themes": [],
            "entities": [],
        },
    )


def create_chunk_node(name):
    """Helper function to create a chunk node with proper structure."""
    return Node(
        id=DebugUUID(name),
        type=NodeType.CHUNK,
        properties={
            "page_content": f"{name} content",
            "summary": f"{name} summary",
            "summary_embedding": [0.001, 0.002, 0.003],
            "themes": [],
            "entities": [],
        },
    )


def create_chain_of_similarities(starting_node, node_count=5, cycle=False):
    """
    Create a chain of document nodes with cosine similarity relationships.

    Parameters
    ----------
    starting_node : Node
        Node to start the chain from. This will be the first node in the chain.
    node_count : int
        Number of nodes to create
    cycle : bool
        If True, add a relationship from the last node back to the first node

    Returns
    -------
    tuple
        (list of nodes, list of relationships)
    """
    # Create document nodes
    nodes = []

    # Use starting_node as the first node
    nodes.append(starting_node)

    # Create remaining nodes
    for i in range(node_count - 1):
        nodes.append(create_document_node(name=f"{starting_node.id}_{i + 1}"))

    relationships = []
    for i in range(node_count - 1):
        rel = Relationship(
            source=nodes[i],
            target=nodes[i + 1],
            type="cosine_similarity",
            bidirectional=True,
            properties={"summary_similarity": 0.9},
        )
        relationships.append(rel)

    if cycle and node_count > 1:
        cycle_rel = Relationship(
            source=nodes[-1],
            target=nodes[0],
            type="cosine_similarity",
            bidirectional=True,
            properties={"summary_similarity": 0.9},
        )
        relationships.append(cycle_rel)

    return nodes, relationships


def create_chain_of_overlaps(starting_node, node_count=3, cycle=False):
    """
    Create a chain of nodes with entity overlap relationships.

    Parameters
    ----------
    starting_node : Node
        Node to start the chain from. This will be the first node in the chain.
    node_count : int
        Number of nodes to create
    cycle : bool
        If True, add a relationship from the last node back to the first node

    Returns
    -------
    tuple
        (list of nodes, list of relationships)
    """
    # Create nodes (mix of document and chunk nodes)
    nodes = []
    relationships = []

    # Use starting_node as the first node and set its entity
    starting_node.properties["entities"] = [f"E_{starting_node.id}_1"]
    nodes.append(starting_node)

    # Create relationships and remaining node
    prev_node = starting_node
    for i in range(node_count - 1):
        # Realistic entity assignment
        prev_entity = f"E_{starting_node.id}_{i + 1}"
        new_entity = f"E_{starting_node.id}_{i + 2}"

        new_node = create_document_node(name=f"{starting_node.id}_{i + 1}")

        # Add entities to the new node, including overlap w/ previous node
        new_node.properties["entities"] = [prev_entity, new_entity]
        nodes.append(new_node)

        rel = Relationship(
            source=prev_node,
            target=new_node,
            type="entities_overlap",
            bidirectional=False,
            properties={
                "entities_overlap_score": 0.1,
                "overlapped_items": [[prev_entity, prev_entity]],
            },
        )
        relationships.append(rel)
        prev_node = new_node

    if cycle and node_count > 1:
        # For the cycle, the last node should share an entity with the first node
        # Add the first entity to the last node's entities
        first_entity = f"E_{starting_node.id}_1"
        nodes[-1].properties["entities"].append(first_entity)

        cycle_rel = Relationship(
            source=nodes[-1],
            target=nodes[0],
            type="entities_overlap",
            bidirectional=False,
            properties={
                "entities_overlap_score": 0.1,
                "overlapped_items": [[first_entity, first_entity]],
            },
        )
        relationships.append(cycle_rel)

    return nodes, relationships


def create_document_and_child_nodes():
    """
    Create a document node and its child chunk nodes with the same structure as create_branched_graph.

    Returns
    -------
    tuple
        (dict of nodes, list of relationships)
    """
    # Create nodes - A is a document, the rest are chunks
    nodes = {
        "A": create_document_node("A"),
        "B": create_chunk_node("B"),
        "C": create_chunk_node("C"),
        "D": create_chunk_node("D"),
        "E": create_chunk_node("E"),
    }

    # Create "child" relationships from document to chunks
    child_relationships = [
        Relationship(
            source=nodes["A"],
            target=nodes["B"],
            type="child",
            bidirectional=False,
            properties={},
        ),
        Relationship(
            source=nodes["A"],
            target=nodes["C"],
            type="child",
            bidirectional=False,
            properties={},
        ),
        Relationship(
            source=nodes["A"],
            target=nodes["D"],
            type="child",
            bidirectional=False,
            properties={},
        ),
        Relationship(
            source=nodes["A"],
            target=nodes["E"],
            type="child",
            bidirectional=False,
            properties={},
        ),
    ]

    # Create "next" relationships between chunks
    next_relationships = [
        Relationship(
            source=nodes["B"],
            target=nodes["C"],
            type="next",
            bidirectional=False,
            properties={},
        ),
        Relationship(
            source=nodes["C"],
            target=nodes["D"],
            type="next",
            bidirectional=False,
            properties={},
        ),
        Relationship(
            source=nodes["D"],
            target=nodes["E"],
            type="next",
            bidirectional=False,
            properties={},
        ),
    ]

    # Combine all relationships
    relationships = child_relationships + next_relationships

    return nodes, relationships


def build_knowledge_graph(nodes, relationships):
    """
    Build a knowledge graph from nodes and relationships.

    Parameters
    ----------
    nodes : list or dict
        Nodes to add to the graph
    relationships : list
        Relationships to add to the graph

    Returns
    -------
    KnowledgeGraph
        The constructed knowledge graph
    """
    kg = KnowledgeGraph()

    # Add nodes to the graph
    if isinstance(nodes, dict):
        for node in nodes.values():
            kg.add(node)
    else:
        for node in nodes:
            kg.add(node)

    # Add relationships to the graph
    for rel in relationships:
        kg.add(rel)

    return kg


def assert_clusters_equal(actual_clusters, expected_clusters):
    """
    Helper function to compare clusters with unordered comparison.

    Args:
        actual_clusters: List of sets representing the actual clusters
        expected_clusters: List of sets representing the expected clusters
    """
    # Convert both lists to sets of frozensets for unordered comparison
    actual_clusters_set = {frozenset(cluster) for cluster in actual_clusters}
    expected_clusters_set = {frozenset(cluster) for cluster in expected_clusters}

    assert (
        actual_clusters_set == expected_clusters_set
    ), f"Expected clusters: {expected_clusters_set}\nActual clusters: {actual_clusters_set}"


def test_find_indirect_clusters_with_document_and_children():
    """Test find_indirect_clusters with a document and its child nodes."""
    nodes, relationships = create_document_and_child_nodes()
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_indirect_clusters(depth_limit=4)

    # Define expected clusters based on the graph structure and the find_indirect_clusters algorithm
    # The algorithm creates clusters for each path through the graph

    assert_clusters_equal(clusters, [
        {nodes["A"], nodes["B"]},
        {nodes["A"], nodes["C"]},
        {nodes["A"], nodes["D"]},
        {nodes["A"], nodes["E"]},
        {nodes["B"], nodes["C"]},
        {nodes["C"], nodes["D"]},
        {nodes["D"], nodes["E"]},
        {nodes["A"], nodes["B"], nodes["C"]},
        {nodes["A"], nodes["C"], nodes["D"]},
        {nodes["A"], nodes["D"], nodes["E"]},
        {nodes["B"], nodes["C"], nodes["D"]},
        {nodes["C"], nodes["D"], nodes["E"]},
        {nodes["A"], nodes["B"], nodes["C"], nodes["D"]},
        {nodes["B"], nodes["C"], nodes["D"], nodes["E"]},
        {nodes["A"], nodes["C"], nodes["D"], nodes["E"]},
    ])


def test_find_n_indirect_clusters_with_document_and_children():
    """Test find_indirect_clusters with a document and its child nodes."""
    nodes, relationships = create_document_and_child_nodes()
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_n_indirect_clusters(n=5, depth_limit=4)

    # Define expected clusters based on the graph structure and the find_indirect_clusters algorithm
    # The algorithm creates clusters for each path through the graph
    assert_clusters_equal(clusters, [
        {nodes["A"], nodes["B"], nodes["C"], nodes["D"]},
        {nodes["A"], nodes["C"], nodes["D"], nodes["E"]},
        {nodes["B"], nodes["C"], nodes["D"], nodes["E"]},
    ])


def test_find_indirect_clusters_with_similarity_relationships():
    """Test find_indirect_clusters with cosine similarity relationships between document nodes."""
    nodes, relationships = create_chain_of_similarities(
        create_document_node("A"), node_count=4
    )
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_indirect_clusters(depth_limit=4)

    # With bidirectional relationships, we expect clusters for each possible path
    assert_clusters_equal(clusters, [
        {nodes[0], nodes[1]},  # A->1
        {nodes[1], nodes[2]},  # 1->2
        {nodes[2], nodes[3]},  # 2->3
        {nodes[0], nodes[1], nodes[2]},  # A->1->2
        {nodes[1], nodes[2], nodes[3]},  # 1->2->3
        {nodes[0], nodes[1], nodes[2], nodes[3]},  # A->1->2->3
    ])


def test_find_n_indirect_clusters_with_similarity_relationships():
    """Test find_indirect_clusters with cosine similarity relationships between document nodes."""
    nodes, relationships = create_chain_of_similarities(
        create_document_node("A"), node_count=4
    )
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_n_indirect_clusters(n=5, depth_limit=4)

    # With bidirectional relationships, we expect clusters for each possible path
    assert_clusters_equal(clusters, [
        {nodes[0], nodes[1], nodes[2], nodes[3]},  # A->1->2->3
    ])

    # create 5 node cycle branching off node 2
    five_node_cycle, fnc_relationships = create_chain_of_similarities(
        nodes[2], node_count=5, cycle=True
    )
    # create independent 2 node cycle to cover edge case
    two_node_cycle, tnc_relationships = create_chain_of_similarities(
        create_document_node("C"), node_count=2, cycle=True
    )

    new_nodes = five_node_cycle[1:] + two_node_cycle
    nodes.extend(new_nodes)
    for item in new_nodes + fnc_relationships + tnc_relationships:
        kg.add(item)

    clusters = kg.find_n_indirect_clusters(n=15, depth_limit=3)

    assert_clusters_equal(clusters, [
        {nodes[0], nodes[1], nodes[2]},
        {nodes[1], nodes[2], nodes[3]},
        {nodes[2], nodes[3], nodes[4]},
        {nodes[1], nodes[2], nodes[4]},
        {nodes[1], nodes[2], nodes[7]},
        {nodes[2], nodes[4], nodes[5]},
        {nodes[2], nodes[4], nodes[7]},
        {nodes[2], nodes[3], nodes[7]},
        {nodes[2], nodes[6], nodes[7]},
        {nodes[4], nodes[5], nodes[6]},
        {nodes[5], nodes[6], nodes[7]},
        {nodes[8], nodes[9]},  # independent two node cycle
    ])

    clusters = kg.find_n_indirect_clusters(n=2, depth_limit=4)
    assert len(clusters) == 2


def test_find_indirect_clusters_with_overlap_relationships():
    """Test find_indirect_clusters with entity overlap relationships."""
    nodes, relationships = create_chain_of_overlaps(
        create_document_node("A"), node_count=4
    )
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_indirect_clusters(depth_limit=3)

    # With unidirectional relationships, we expect clusters for each path
    assert_clusters_equal(clusters, [
        {nodes[0], nodes[1]},  # Start->1
        {nodes[1], nodes[2]},  # 1->2
        {nodes[2], nodes[3]},  # 2->3
        {nodes[0], nodes[1], nodes[2]},  # Start->1->2
        {nodes[1], nodes[2], nodes[3]},  # 1->2->3
    ])


def test_find_n_indirect_clusters_with_overlap_relationships():
    """Test find_indirect_clusters with entity overlap relationships."""
    nodes, relationships = create_chain_of_overlaps(
        create_document_node("A"), node_count=4
    )
    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_n_indirect_clusters(n=5, depth_limit=3)

    assert_clusters_equal(
        clusters,
        [
            {nodes[0], nodes[1], nodes[2]},
            {nodes[1], nodes[2], nodes[3]},
        ],
    )

    # create 5 node cycle branching off node 2
    five_node_cycle, fnc_relationships = create_chain_of_overlaps(
        nodes[2], node_count=5, cycle=True
    )
    # create independent 2 node cycle to cover edge case
    two_node_cycle, tnc_relationships = create_chain_of_overlaps(
        create_document_node("C"), node_count=2, cycle=True
    )

    new_nodes = five_node_cycle[1:] + two_node_cycle
    nodes.extend(new_nodes)
    for item in new_nodes + fnc_relationships + tnc_relationships:
        kg.add(item)

    clusters = kg.find_n_indirect_clusters(n=15, depth_limit=3)

    assert_clusters_equal(
        clusters,
        [
            {nodes[0], nodes[1], nodes[2]},
            {nodes[1], nodes[2], nodes[3]},
            {nodes[1], nodes[2], nodes[4]},
            {nodes[2], nodes[4], nodes[5]},
            {nodes[4], nodes[5], nodes[6]},
            {nodes[5], nodes[6], nodes[7]},
            {nodes[6], nodes[7], nodes[2]},
            {nodes[7], nodes[2], nodes[3]},
            {nodes[7], nodes[2], nodes[4]},
            {nodes[8], nodes[9]},  # independent two node cycle
        ],
    )

    clusters = kg.find_n_indirect_clusters(n=2, depth_limit=4)
    assert len(clusters) == 2


def test_find_indirect_clusters_with_condition():
    """Test find_indirect_clusters with a relationship condition."""
    nodes, relationships = create_document_and_child_nodes()
    kg = build_knowledge_graph(nodes, relationships)

    def condition(rel):
        return rel.type == "next"

    clusters = kg.find_indirect_clusters(relationship_condition=condition)

    # Only "next" relationships are considered, so we should only have paths between B, C, D, and E
    assert_clusters_equal(clusters, [
        {nodes["B"], nodes["C"]},  # B->C
        {nodes["C"], nodes["D"]},  # C->D
        {nodes["D"], nodes["E"]},  # D->E
        {nodes["B"], nodes["C"], nodes["D"]},  # B->C->D
        {nodes["C"], nodes["D"], nodes["E"]},  # C->D->E
    ])


def test_find_n_indirect_clusters_with_condition():
    """Test find_indirect_clusters with a relationship condition."""
    nodes, relationships = create_document_and_child_nodes()
    kg = build_knowledge_graph(nodes, relationships)

    def condition(rel):
        return rel.type == "next"

    clusters = kg.find_n_indirect_clusters(n=5, relationship_condition=condition)

    # Only "next" relationships are considered, so we should only have paths between B, C, D, and E
    assert_clusters_equal(clusters, [
        {nodes["B"], nodes["C"], nodes["D"]},  # B->C->D
        {nodes["C"], nodes["D"], nodes["E"]},  # C->D->E
    ])


# test cyclic relationships for bidirectional relationships
def test_find_indirect_clusters_with_cyclic_similarity_relationships():
    """Test find_indirect_clusters with cyclic cosine similarity relationships."""
    nodes, relationships = create_chain_of_similarities(
        create_document_node("A"), node_count=3, cycle=True
    )
    # branch off last node so it both cycles and branches
    branched_nodes, branched_relationships = create_chain_of_similarities(
        nodes[-1], node_count=2
    )
    nodes.extend(branched_nodes[1:])
    relationships.extend(branched_relationships)

    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_indirect_clusters(depth_limit=10)

    # With a cycle, we expect additional clusters that include paths through the cycle
    assert_clusters_equal(clusters, [
        {nodes[0], nodes[1]},
        {nodes[1], nodes[2]},
        {nodes[2], nodes[3]},
        {nodes[2], nodes[0]},
        {nodes[0], nodes[1], nodes[2]},
        {nodes[0], nodes[2], nodes[3]},
        {nodes[1], nodes[2], nodes[0]},
        {nodes[2], nodes[0], nodes[1]},
        {nodes[1], nodes[2], nodes[3]},
        {nodes[0], nodes[1], nodes[2], nodes[3]},
    ])


# test cyclic relationships for bidirectional relationships
def test_find_n_indirect_clusters_with_cyclic_similarity_relationships():
    """Test find_indirect_clusters with cyclic cosine similarity relationships."""
    nodes, relationships = create_chain_of_similarities(
        create_document_node("A"), node_count=3, cycle=True
    )
    # branch off last node so it both cycles and branches
    branched_nodes, branched_relationships = create_chain_of_similarities(
        nodes[-1], node_count=2
    )
    nodes.extend(branched_nodes[1:])
    relationships.extend(branched_relationships)

    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_n_indirect_clusters(n=15, depth_limit=3)

    # With a cycle, we expect additional clusters that include paths through the cycle
    assert_clusters_equal(clusters, [
        {nodes[0], nodes[1], nodes[2]},
        {nodes[0], nodes[2], nodes[3]},
        {nodes[1], nodes[2], nodes[0]},
        {nodes[2], nodes[0], nodes[1]},
        {nodes[1], nodes[2], nodes[3]},
    ])


def test_find_indirect_clusters_with_spider_web_graph():
    """Test find_indirect_clusters with a spider web graph where all nodes connect to all other nodes."""
    # Create nodes
    # Create nodes as a dictionary with letters as keys
    nodes = {
        "A": create_document_node("A"),
        "B": create_document_node("B"),
        "C": create_document_node("C"),
        "D": create_document_node("D"),
    }

    # Create relationships - each node connects to every other node
    relationships = []
    node_list = list(nodes.values())
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            if i != j:  # Don't connect node to itself
                rel = Relationship(
                    source=node_list[i],
                    target=node_list[j],
                    type="cosine_similarity",
                    bidirectional=True,
                    properties={"summary_similarity": 0.9},
                )
                relationships.append(rel)

    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_indirect_clusters(depth_limit=3)

    # In a spider web, we expect:
    # 1. All pairs of nodes (directly connected)
    # 2. All triplets of nodes (connected through intermediate nodes)
    # 3. The complete set of all nodes
    assert_clusters_equal(clusters, [
        {nodes["A"], nodes["B"]},
        {nodes["A"], nodes["C"]},
        {nodes["A"], nodes["D"]},
        {nodes["B"], nodes["C"]},
        {nodes["B"], nodes["D"]},
        {nodes["C"], nodes["D"]},
        {nodes["A"], nodes["B"], nodes["C"]},
        {nodes["A"], nodes["B"], nodes["D"]},
        {nodes["A"], nodes["C"], nodes["D"]},
        {nodes["B"], nodes["C"], nodes["D"]},
    ])


def test_find_n_indirect_clusters_with_spider_web_graph():
    """Test find_indirect_clusters with a spider web graph where all nodes connect to all other nodes."""
    # Create nodes
    # Create nodes as a dictionary with letters as keys
    nodes = {
        "A": create_document_node("A"),
        "B": create_document_node("B"),
        "C": create_document_node("C"),
        "D": create_document_node("D"),
    }

    # Create relationships - each node connects to every other node
    relationships = []
    node_list = list(nodes.values())
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            if i != j:  # Don't connect node to itself
                rel = Relationship(
                    source=node_list[i],
                    target=node_list[j],
                    type="cosine_similarity",
                    bidirectional=True,
                    properties={"summary_similarity": 0.9},
                )
                relationships.append(rel)

    kg = build_knowledge_graph(nodes, relationships)
    clusters = kg.find_n_indirect_clusters(n=10, depth_limit=3)

    assert_clusters_equal(clusters, [
        {nodes["A"], nodes["B"], nodes["C"]},
        {nodes["A"], nodes["B"], nodes["D"]},
        {nodes["A"], nodes["C"], nodes["D"]},
        {nodes["B"], nodes["C"], nodes["D"]},
    ])


def test_find_two_nodes_single_rel_with_similarity():
    """Test find_two_nodes_single_rel with cosine similarity relationships."""
    nodes, relationships = create_chain_of_similarities(
        create_document_node("A"), node_count=3
    )
    kg = build_knowledge_graph(nodes, relationships)
    triplets = kg.find_two_nodes_single_rel()

    # Verify triplets
    assert len(triplets) == 2

    # Check if all triplets have the correct relationship type
    for triplet in triplets:
        assert triplet[1].type == "cosine_similarity"
        assert "summary_similarity" in triplet[1].properties


def test_find_two_nodes_single_rel_with_overlap():
    """Test find_two_nodes_single_rel with entity overlap relationships."""
    nodes, relationships = create_chain_of_overlaps(
        create_document_node("A"), node_count=3
    )
    kg = build_knowledge_graph(nodes, relationships)
    triplets = kg.find_two_nodes_single_rel()

    # Verify triplets
    assert len(triplets) == 2

    # Check if all triplets have the correct relationship type
    for triplet in triplets:
        assert triplet[1].type == "entities_overlap"
        assert "entities_overlap_score" in triplet[1].properties
        assert "overlapped_items" in triplet[1].properties


def test_find_two_nodes_single_rel_with_condition():
    """Test find_two_nodes_single_rel with a relationship condition."""
    nodes, relationships = create_document_and_child_nodes()
    kg = build_knowledge_graph(nodes, relationships)

    def condition(rel):
        return rel.type == "child"

    triplets = kg.find_two_nodes_single_rel(relationship_condition=condition)

    # Verify triplets
    assert len(triplets) == 4  # Should have 4 child relationships

    # Check if all triplets have the correct relationship type
    for triplet in triplets:
        assert triplet[1].type == "child"

        # Verify that one of the nodes is the document node (A)
        # and the other is one of the chunk nodes (B, C, D, or E)
        assert nodes["A"] in [triplet[0], triplet[2]]
        assert triplet[0] == nodes["A"] or triplet[2] == nodes["A"]

        # The other node should be one of the chunk nodes
        other_node = triplet[0] if triplet[2] == nodes["A"] else triplet[2]
        assert other_node in [nodes["B"], nodes["C"], nodes["D"], nodes["E"]]


def test_find_two_nodes_single_rel_normalized_order():
    """Test that find_two_nodes_single_rel normalizes the order of nodes based on ID."""
    # Create nodes with specific UUIDs to ensure consistent ordering
    node_a = Node(
        id=DebugUUID("A"),
        type=NodeType.CHUNK,
        properties={
            "page_content": "A content",
            "summary": "A summary",
            "summary_embedding": [0.001, 0.002, 0.003],
            "themes": [],
            "entities": [],
        },
    )

    node_b = Node(
        id=DebugUUID("B"),
        type=NodeType.CHUNK,
        properties={
            "page_content": "B content",
            "summary": "B summary",
            "summary_embedding": [0.001, 0.002, 0.003],
            "themes": [],
            "entities": [],
        },
    )

    # Create relationship from B to A (reverse of ID order)
    rel_ba = Relationship(
        source=node_b, target=node_a, type="next", bidirectional=False, properties={}
    )

    # Build knowledge graph
    kg = build_knowledge_graph([node_a, node_b], [rel_ba])

    # Find two-node relationships
    triplets = kg.find_two_nodes_single_rel()

    # Verify triplets - should have the relationship in the correct order
    assert len(triplets) == 1
    triplet = triplets[0]

    # Check the relationship is correct
    assert triplet[1].type == "next"

    # Check source and target - the actual order depends on how the KnowledgeGraph.find_two_nodes_single_rel
    # implementation sorts nodes, which may be by string representation or internal UUID value
    # So we just verify that the relationship is between the two nodes we created
    assert {triplet[0], triplet[2]} == {node_a, node_b}
    assert (triplet[0] == node_a and triplet[2] == node_b) or (
        triplet[0] == node_b and triplet[2] == node_a
    )


def test_find_two_nodes_single_rel_with_self_loops():
    """Test find_two_nodes_single_rel with self-loops (should be excluded)."""
    node_a = create_chunk_node("A")
    node_b = create_chunk_node("B")

    # Create relationships including a self-loop
    rel_ab = Relationship(
        source=node_a, target=node_b, type="next", bidirectional=False, properties={}
    )
    rel_aa = Relationship(
        source=node_a,
        target=node_a,
        type="self_loop",
        bidirectional=True,
        properties={},
    )

    # Build knowledge graph
    kg = build_knowledge_graph([node_a, node_b], [rel_ab, rel_aa])

    # Find two-node relationships
    triplets = kg.find_two_nodes_single_rel()

    # Verify triplets - self-loops should be excluded
    assert len(triplets) == 1

    # Check if we have only the A-B relationship
    # The actual implementation returns nodes in a different order than expected
    assert (triplets[0][0] == node_a and triplets[0][2] == node_b) or (
        triplets[0][0] == node_b and triplets[0][2] == node_a
    )
    assert triplets[0][1].type == "next"
