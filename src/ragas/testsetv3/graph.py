from graphene import ObjectType, String, List, Field, Int


class Relationship(ObjectType):
    """Represents a directed relationship between two nodes in a graph.

    Attributes:
        source (Node): The source node of the relationship.
        target (Node): The target node of the relationship.
        label (str): The label of the relationship.
        properties (dict): Additional properties associated with the relationship.
    """

    source = Field(lambda: Node)
    target = Field(lambda: Node)
    label = String()
    properties = Field(lambda: dict)


class Node(ObjectType):
    """Represents a node in a graph with associated properties.

    Attributes:
        id (Union[str, int]): A unique identifier for the node.
        label (str): The label or label of the node, default is "Node".
        properties (dict): Additional properties and metadata associated with the node.
    """

    id = String()
    label = String()
    properties = Field(lambda: dict)
    relationships = List(Relationship)