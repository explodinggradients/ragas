from graphene import ObjectType, String, List, Field, JSONString, Enum, Schema, Argument
import uuid

class NodeType(Enum):
    DOC = "doc"
    CHUNK = "chunk"


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
    properties = Field(JSONString)


class Node(ObjectType):
    """Represents a node in a graph with associated properties.

    Attributes:
        id (Union[str, int]): A unique identifier for the node.
        label (str): The label or label of the node, default is "Node".
        properties (dict): Additional properties and metadata associated with the node.
    """

    id = String(required=True)
    label = Field(NodeType)
    properties = Field(JSONString)
    relationships = List(lambda: Relationship)
    
    def __init__(self, id=None, **kwargs):
        if id is None:
            id = str(uuid.uuid4())
        super().__init__(id=id, **kwargs)


class Query(ObjectType):
    node = Field(Node, id=String(required=True))
    nodes_by_label = List(Node, label=Argument(NodeType, required=True))

    def resolve_node(parent, info, id):
        nodes = info.context.get('nodes', [])
        for node in nodes:
            if node.id == id:
                return node
        return None

    def resolve_nodes_by_label(parent, info, label):
        nodes = info.context.get('nodes', [])
        return [node for node in nodes if node.label == label]
    
    
    
schema = Schema(query=Query)
    